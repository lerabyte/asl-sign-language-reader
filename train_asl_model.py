from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ======================
# SETTINGS
# ======================

DATA_DIR = Path("asl")  # change this only if your asl folder is somewhere else
OUTPUT_DIR = Path("asl_model_output")

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

VALIDATION_SPLIT = 0.20

EPOCHS_HEAD = 8
EPOCHS_FINE_TUNE = 8

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# ======================
# FOLDER HELPERS
# ======================

def has_class_folders(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    folders = [p for p in path.iterdir() if p.is_dir()]
    if len(folders) < 5:
        return False

    folders_with_images = 0
    for folder in folders:
        if any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir()):
            folders_with_images += 1

    return folders_with_images >= 5


def find_train_dir(base: Path) -> Path:
    candidates = [
        base / "asl_alphabet_train" / "asl_alphabet_train",
        base / "asl_alphabet_train",
        base / "train",
        base,
    ]

    for candidate in candidates:
        if has_class_folders(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find your training folders. Expected something like:\n"
        "asl/asl_alphabet_train/asl_alphabet_train/A/*.jpg"
    )


def find_test_dir(base: Path):
    candidates = [
        base / "asl_alphabet_test" / "asl_alphabet_test",
        base / "asl_alphabet_test",
        base / "test",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            image_files = [
                p for p in candidate.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
            if image_files:
                return candidate

    return None


# ======================
# MODEL
# ======================

def make_augmentation():
    return keras.Sequential(
        [
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.12),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_custom_cnn(num_classes: int):
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")

    x = make_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters, dropout in [(32, 0.10), (64, 0.15), (128, 0.20), (256, 0.25)]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(384, activation="relu")(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="letter")(x)

    model = keras.Model(inputs, outputs, name="asl_custom_cnn")
    return model, None, "custom_cnn"


def build_model(num_classes: int):
    """
    Best option: MobileNetV2 transfer learning.
    Backup option: custom CNN, used automatically if ImageNet weights cannot load.
    """
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet",
        )
    except Exception as e:
        print("Could not load pretrained MobileNetV2 weights.")
        print("Using custom CNN instead.")
        print("Reason:", e)
        return build_custom_cnn(num_classes)

    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")

    x = make_augmentation()(inputs)

    # Same preprocessing MobileNetV2 expects, but save/load-safe.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_rescale")(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="letter")(x)

    model = keras.Model(inputs, outputs, name="asl_mobilenetv2")
    return model, base_model, "mobilenetv2_imagenet"


def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ======================
# TINY TEST FOLDER
# ======================

def label_from_test_filename(path: Path, class_names):
    """
    Examples:
    A_test.jpg -> A
    space_test.jpg -> space
    nothing_test.jpg -> nothing
    """
    stem = path.stem

    if stem.endswith("_test"):
        stem = stem[:-5]

    if stem in class_names:
        return stem

    lower_stem = stem.lower()
    for class_name in sorted(class_names, key=len, reverse=True):
        lower_name = class_name.lower()
        if lower_stem == lower_name or lower_stem.startswith(lower_name + "_"):
            return class_name

    return None


def run_tiny_test(model, test_dir: Path, class_names):
    if test_dir is None:
        print("No tiny test folder found, so skipping final test.")
        return

    test_images = sorted(
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    if not test_images:
        print("Tiny test folder was found, but it has no images.")
        return

    print()
    print("Tiny test results:")
    print("-" * 70)

    correct = 0
    total = 0

    for image_path in test_images:
        true_label = label_from_test_filename(image_path, class_names)
        if true_label is None:
            print(f"Skipping {image_path.name}: label not recognized.")
            continue

        image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        array = tf.keras.utils.img_to_array(image)
        array = array[None, ...]

        probs = model.predict(array, verbose=0)[0]
        pred_index = int(probs.argmax())
        pred_label = class_names[pred_index]
        confidence = float(probs[pred_index])

        total += 1
        if pred_label == true_label:
            correct += 1

        print(
            f"{image_path.name:25s} "
            f"true={true_label:10s} "
            f"pred={pred_label:10s} "
            f"conf={confidence:.2%}"
        )

    if total > 0:
        print("-" * 70)
        print(f"Tiny test accuracy: {correct}/{total} = {correct / total:.2%}")


def get_val_accuracy(model, val_ds):
    loss_value, accuracy = model.evaluate(val_ds, verbose=0)
    return float(accuracy)


# ======================
# TRAIN
# ======================

def main():
    train_dir = find_train_dir(DATA_DIR)
    test_dir = find_test_dir(DATA_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Training folder:", train_dir)
    print("Test folder:", test_dir if test_dir else "not found")
    print()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)

    print()
    print("Classes:")
    print(class_names)
    print("Number of classes:", num_classes)
    print()

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    model, base_model, model_type = build_model(num_classes)

    metadata = {
        "class_names": class_names,
        "image_size": list(IMG_SIZE),
        "model_type": model_type,
    }

    with open(OUTPUT_DIR / "asl_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    loss = keras.losses.SparseCategoricalCrossentropy()

    best_model = model
    best_val_accuracy = -1.0

    if base_model is None:
        # Custom CNN trains from scratch, so let it use the full epoch count.
        total_epochs = EPOCHS_HEAD + EPOCHS_FINE_TUNE

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            callbacks=make_callbacks(),
        )

        best_val_accuracy = get_val_accuracy(model, val_ds)
        best_model = model

    else:
        # Stage 1: train only the new classifier head.
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
            metrics=["accuracy"],
        )

        print()
        print("Stage 1: training classifier head")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_HEAD,
            callbacks=make_callbacks(),
        )

        stage1_accuracy = get_val_accuracy(model, val_ds)
        stage1_path = OUTPUT_DIR / "stage1_model.keras"
        model.save(stage1_path)

        best_val_accuracy = stage1_accuracy
        best_model = model

        print(f"Stage 1 validation accuracy: {stage1_accuracy:.2%}")

        # Stage 2: fine-tune the last part of MobileNetV2.
        print()
        print("Stage 2: fine-tuning MobileNetV2")

        base_model.trainable = True

        for layer in base_model.layers[:-35]:
            layer.trainable = False

        # Keep BatchNorm frozen during fine-tuning.
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=loss,
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINE_TUNE,
            callbacks=make_callbacks(),
        )

        stage2_accuracy = get_val_accuracy(model, val_ds)
        print(f"Stage 2 validation accuracy: {stage2_accuracy:.2%}")

        if stage2_accuracy >= stage1_accuracy:
            best_val_accuracy = stage2_accuracy
            best_model = model
        else:
            print("Stage 2 did not beat Stage 1, so keeping Stage 1 model.")
            best_model = keras.models.load_model(stage1_path)

    best_path = OUTPUT_DIR / "best_asl_model.keras"
    final_path = OUTPUT_DIR / "final_asl_model.keras"

    best_model.save(best_path)
    best_model.save(final_path)

    print()
    print("Saved:")
    print("Best model: ", best_path)
    print("Final model:", final_path)
    print("Metadata:   ", OUTPUT_DIR / "asl_metadata.json")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")

    run_tiny_test(best_model, test_dir, class_names)


if __name__ == "__main__":
    main()
