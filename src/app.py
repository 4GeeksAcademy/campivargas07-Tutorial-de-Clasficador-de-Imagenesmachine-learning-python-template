"""Entrenamiento reproducible de un clasificador de gatos y perros."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (classification_report, confusion_matrix,
							 precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/raw/dogs-vs-cats/train"
DEFAULT_MANIFEST = ROOT / "data/processed/splits.csv"
DEFAULT_MODELS = ROOT / "models"
CLASS_TO_INDEX = {"cat": 0, "dog": 1}
SEED = 42
IMAGE_SIZE = (200, 200)
IMAGE_PATTERN = re.compile(r"^(cat|dog)\.(\d+)\.jpg$", re.IGNORECASE)
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def audit_dataset(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	"""Inspect images and group exact duplicates; return usable rows and audit."""
	if not data_dir.is_dir():
		raise FileNotFoundError(f"No existe la carpeta de imágenes: {data_dir}")

	rows: list[dict[str, Any]] = []
	excluded: list[dict[str, str]] = []
	dimensions: Counter[str] = Counter()
	modes: Counter[str] = Counter()
	hashes: dict[str, list[int]] = defaultdict(list)
	class_counts: Counter[str] = Counter()
	for path in sorted(data_dir.iterdir(), key=lambda item: item.name.lower()):
		if not path.is_file():
			continue
		match = IMAGE_PATTERN.fullmatch(path.name)
		if not match:
			excluded.append({"filename": path.name, "reason": "nombre no reconocido"})
			continue
		label = match.group(1).lower()
		try:
			digest = _sha256(path)
			with Image.open(path) as image:
				image.verify()
			with Image.open(path) as image:
				modes[image.mode] += 1
				dimensions[f"{image.width}x{image.height}"] += 1
				if image.mode not in ("RGB", "RGBA", "L"):
					excluded.append({"filename": path.name, "reason": f"modo no soportado: {image.mode}"})
					continue
		except (OSError, UnidentifiedImageError, ValueError) as error:
			excluded.append({"filename": path.name, "reason": f"imagen inválida: {error}"})
			continue
		row = {
			"filename": path.name,
			"filepath": os.path.relpath(path.resolve(), ROOT),
			"label": label,
			"class_index": CLASS_TO_INDEX[label],
			"sha256": digest,
		}
		hashes[digest].append(len(rows))
		rows.append(row)
		class_counts[label] += 1

	# A hash is the group key, so every exact duplicate is assigned together.
	for digest, indices in hashes.items():
		for index in indices:
			rows[index]["group_id"] = digest

	duplicate_groups = [members for members in hashes.values() if len(members) > 1]
	audit = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"data_dir": data_dir.resolve().as_posix(),
		"num_images_usable": len(rows),
		"class_counts": dict(sorted(class_counts.items())),
		"dimensions_top_10": dimensions.most_common(10),
		"image_modes": dict(modes),
		"unique_sha256": len(hashes),
		"duplicate_groups": len(duplicate_groups),
		"duplicate_files_beyond_first": sum(len(group) - 1 for group in duplicate_groups),
		"excluded_count": len(excluded),
		"excluded_files": excluded,
		"class_to_index": CLASS_TO_INDEX,
		"split_ratios": SPLIT_RATIOS,
		"seed": SEED,
	}
	return rows, audit


def assign_splits(rows: list[dict[str, Any]], seed: int = SEED) -> None:
	"""Mutate rows with deterministic, approximately stratified group splits."""
	by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		by_group[row["group_id"]].append(row)
	rng = random.Random(seed)
	entries = list(by_group.values())
	rng.shuffle(entries)
	labels = sorted({row["label"] for row in rows})
	target_counts = {
		label: {split: SPLIT_RATIOS[split] * sum(row["label"] == label for row in rows)
				for split in SPLIT_RATIOS}
		for label in labels
	}
	assigned = {label: Counter() for label in labels}
	for members in entries:
		group_counts = Counter(row["label"] for row in members)

		def split_cost(split: str) -> tuple[float, int]:
			# Greedily place the entire group where its class counts best reduce
			# the current deficits; a cross-label duplicate can never leak.
			cost = sum(
				(assigned[label][split] + group_counts[label] - target_counts[label][split]) ** 2
				- (assigned[label][split] - target_counts[label][split]) ** 2
				for label in labels
			)
			return cost, list(SPLIT_RATIOS).index(split)

		split = min(SPLIT_RATIOS, key=split_cost)
		for row in members:
			row["split"] = split
		for label, count in group_counts.items():
			assigned[label][split] += count


def create_manifest(data_dir: Path = DEFAULT_DATA, manifest_path: Path = DEFAULT_MANIFEST,
					seed: int = SEED) -> dict[str, Any]:
	rows, audit = audit_dataset(data_dir)
	if not rows:
		raise ValueError(f"No se encontró ninguna imagen válida en {data_dir}")
	assign_splits(rows, seed)
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	fields = ["filename", "filepath", "label", "class_index", "sha256", "group_id", "split"]
	with manifest_path.open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)
	audit["seed"] = seed
	audit["split_counts"] = {
		split: dict(sorted(Counter(row["label"] for row in rows if row["split"] == split).items()))
		for split in SPLIT_RATIOS
	}
	audit["manifest"] = manifest_path.resolve().as_posix()
	audit_path = manifest_path.with_name("audit.json")
	audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
	exclusion_path = manifest_path.with_name("excluded_files.csv")
	with exclusion_path.open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=["filename", "reason"])
		writer.writeheader()
		writer.writerows(audit["excluded_files"])
	return audit


def _tensorflow() -> Any:
	try:
		import tensorflow as tf
	except ImportError as error:
		raise RuntimeError("TensorFlow no está instalado. Instala requirements.txt en el entorno activo.") from error
	return tf


def set_reproducibility(seed: int = SEED) -> None:
	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)
	np.random.seed(seed)
	tf = _tensorflow()
	tf.keras.utils.set_random_seed(seed)
	try:
		tf.config.experimental.enable_op_determinism()
	except (AttributeError, RuntimeError):
		pass


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
	with manifest_path.open(newline="", encoding="utf-8") as stream:
		rows = list(csv.DictReader(stream))
	if not rows:
		raise ValueError(f"El manifiesto está vacío: {manifest_path}")
	seen: dict[str, str] = {}
	for row in rows:
		if row["label"] not in CLASS_TO_INDEX or int(row["class_index"]) != CLASS_TO_INDEX[row["label"]]:
			raise ValueError(f"Mapeo de clase inválido en {row['filename']}")
		previous = seen.setdefault(row["group_id"], row["split"])
		if previous != row["split"]:
			raise ValueError(f"Fuga entre particiones para grupo {row['group_id']}")
		image_path = Path(row["filepath"])
		if not image_path.is_absolute():
			image_path = ROOT / image_path
		row["filepath"] = image_path.resolve().as_posix()
		if not image_path.is_file():
			raise FileNotFoundError(str(image_path))
	return rows


def make_model() -> Any:
	tf = _tensorflow()
	keras = tf.keras
	augmentation = keras.Sequential(
		[keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.08), keras.layers.RandomZoom(0.10)],
		name="augmentation",
	)
	inputs = keras.Input(shape=(200, 200, 3), name="image_rgb_200")
	x = augmentation(inputs)
	backbone = keras.applications.EfficientNetB0(
		include_top=False, weights=None, input_shape=(200, 200, 3), pooling=None,
	)
	backbone.trainable = True
	x = backbone(x)
	x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
	x = keras.layers.Dense(128, activation="relu", name="dense_128")(x)
	outputs = keras.layers.Dense(2, activation="softmax", name="class_probabilities")(x)
	model = keras.Model(inputs=inputs, outputs=outputs, name="dogs_cats_efficientnetb0")
	return model


def _dataset(tf: Any, rows: list[dict[str, Any]], batch_size: int, training: bool,
			 seed: int) -> Any:
	paths = [row["filepath"] for row in rows]
	labels = [int(row["class_index"]) for row in rows]
	dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
	if training:
		dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)

	def decode(path: Any, label: Any) -> tuple[Any, Any]:
		image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
		image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
		# EfficientNetB0 de Keras incorpora Rescaling(1/255); no se reescala aquí.
		image = tf.cast(image, tf.float32)
		return image, label

	dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
	options = tf.data.Options()
	options.experimental_deterministic = True
	return dataset.with_options(options)


def train_model(manifest_path: Path = DEFAULT_MANIFEST, models_dir: Path = DEFAULT_MODELS,
				batch_size: int = 16, epochs: int = 30, learning_rate: float = 1e-3,
				seed: int = SEED) -> dict[str, Any]:
	tf = _tensorflow()
	set_reproducibility(seed)
	rows = load_manifest(manifest_path)
	partitions = {name: [row for row in rows if row["split"] == name]
				  for name in ("train", "validation", "test")}
	missing = [name for name, subset in partitions.items() if not subset]
	if missing:
		raise ValueError(f"Particiones vacías en el manifiesto: {missing}")
	train_ds = _dataset(tf, partitions["train"], batch_size, True, seed)
	val_ds = _dataset(tf, partitions["validation"], batch_size, False, seed)
	test_ds = _dataset(tf, partitions["test"], batch_size, False, seed)

	models_dir.mkdir(parents=True, exist_ok=True)
	checkpoint = models_dir / "best.keras"
	model = make_model()
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
				  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", mode="min",
										   save_best_only=True, verbose=1),
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=6,
										 restore_best_weights=True, verbose=1),
		tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.3,
											 patience=2, min_lr=1e-6, verbose=1),
	]
	history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
						verbose=1)
	if not checkpoint.is_file():
		raise RuntimeError("No se generó el checkpoint de validación; no se evaluará test.")
	# Selección del mejor checkpoint exclusivamente por val_loss. Test se consulta una sola vez aquí.
	best_model = tf.keras.models.load_model(checkpoint)
	# Materialize prediction and labels in one traversal: the held-out test split
	# must not be repeatedly queried during selection or reporting.
	probabilities = best_model.predict(test_ds, verbose=1)
	y_true = np.asarray([int(row["class_index"]) for row in partitions["test"]])
	y_pred = probabilities.argmax(axis=1)
	test_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, probabilities).numpy()
	test_loss = float(np.mean(test_losses))
	test_accuracy = float(np.mean(y_true == y_pred))
	precision, recall, f1, _ = precision_recall_fscore_support(
		y_true, y_pred, labels=[0, 1], zero_division=0,
	)
	metrics = {
		"test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
		"test_precision": precision.tolist(), "test_recall": recall.tolist(), "test_f1": f1.tolist(),
		"test_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
		"test_classification_report": classification_report(
			y_true, y_pred, labels=[0, 1], target_names=["cat", "dog"], output_dict=True, zero_division=0,
		),
		"test_examples": len(y_true), "test_was_evaluated_once": True,
	}
	metadata = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"architecture": "EfficientNetB0(include_top=False, weights=None, input_shape=(200, 200, 3)); trainable backbone; GAP; Dense(128,relu); Dense(2,softmax)",
		"class_to_index": CLASS_TO_INDEX, "input": {"shape": [200, 200, 3], "color": "RGB", "pixel_range": "0-255 (EfficientNet internal rescaling)"},
		"loss": "sparse_categorical_crossentropy", "seed": seed,
		"training_config": {"epochs_requested": epochs, "epochs_completed": len(history.history["loss"]),
							"batch_size": batch_size, "initial_learning_rate": learning_rate,
							"augmentation": ["RandomFlip(horizontal)", "RandomRotation(0.08)", "RandomZoom(0.10)"],
							"checkpoint_selection": "minimum val_loss", "early_stopping_patience": 6,
							"reduce_lr_on_plateau": {"factor": 0.3, "patience": 2}},
		"best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
		"best_validation_loss": float(min(history.history["val_loss"])),
		"validation_history": {key: [float(value) for value in values]
								for key, values in history.history.items()},
		"test_metrics": metrics,
	}
	# Keep the selected complete Keras model as the canonical artifact.
	final_model_path = models_dir / "dogs_cats_efficientnetb0.keras"
	best_model.save(final_model_path)
	metadata_path = models_dir / "metadata.json"
	metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
	history_path = models_dir / "history.json"
	history_path.write_text(json.dumps(metadata["validation_history"], indent=2), encoding="utf-8")
	verify_reload_equivalence(tf, best_model, final_model_path, val_ds, models_dir)
	print(json.dumps({"model": str(final_model_path), "metadata": str(metadata_path), "test_metrics": metrics},
					 indent=2, ensure_ascii=False))
	return metadata


def verify_reload_equivalence(tf: Any, trained_model: Any, model_path: Path,
							  dataset: Any, models_dir: Path,
							  tolerance: float = 1e-6) -> None:
	images, _ = next(iter(dataset.take(1)))
	before = trained_model.predict(images, verbose=0)
	reloaded = tf.keras.models.load_model(model_path)
	after = reloaded.predict(images, verbose=0)
	max_difference = float(np.max(np.abs(before - after)))
	if not np.allclose(before, after, rtol=0.0, atol=tolerance):
		raise AssertionError(f"Predicciones tras recarga difieren (máx={max_difference})")
	result = {"equivalent": True, "atol": tolerance, "max_abs_difference": max_difference,
			  "examples_checked": int(images.shape[0])}
	(models_dir / "reload_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def predict_image(model_path: Path, image_path: Path) -> dict[str, Any]:
	tf = _tensorflow()
	model = tf.keras.models.load_model(model_path)
	image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
	image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
	probabilities = model.predict(tf.expand_dims(tf.cast(image, tf.float32), 0), verbose=0)[0]
	index = int(np.argmax(probabilities))
	class_name = {index: name for name, index in CLASS_TO_INDEX.items()}[index]
	return {"image": str(image_path), "class": class_name,
			"probabilities": {name: float(probabilities[class_index])
							  for name, class_index in CLASS_TO_INDEX.items()}}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	commands = parser.add_subparsers(dest="command", required=True)
	audit = commands.add_parser("audit", help="auditar imágenes y crear manifiesto reproducible")
	audit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
	audit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
	audit.add_argument("--seed", type=int, default=SEED)
	train = commands.add_parser("train", help="entrenar, seleccionar por validación y evaluar test")
	train.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
	train.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
	train.add_argument("--batch-size", type=int, default=16)
	train.add_argument("--epochs", type=int, default=30)
	train.add_argument("--learning-rate", type=float, default=1e-3)
	train.add_argument("--seed", type=int, default=SEED)
	predict = commands.add_parser("predict", help="clasificar una imagen")
	predict.add_argument("image", type=Path)
	predict.add_argument("--model", type=Path, default=DEFAULT_MODELS / "dogs_cats_efficientnetb0.keras")
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	if args.command == "audit":
		result = create_manifest(args.data_dir, args.manifest, args.seed)
		print(json.dumps(result, indent=2, ensure_ascii=False))
	elif args.command == "train":
		train_model(args.manifest, args.models_dir, args.batch_size, args.epochs,
					args.learning_rate, args.seed)
	elif args.command == "predict":
		print(json.dumps(predict_image(args.model, args.image), indent=2, ensure_ascii=False))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
