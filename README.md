# TinyML Iris Classifier on ESP32

Train a small neural network on the Iris dataset, convert it to TensorFlow Lite, and run inference on an ESP32 with TensorFlow Lite Micro. The firmware classifies a flower from four measurements and prints the predicted class and probabilities over serial every five seconds.

The trained model is embedded in [`include/iris_model.h`](include/iris_model.h), so you can build and run the firmware without installing TensorFlow or retraining the model.

## How it works

```text
scikit-learn Iris dataset
        |
        v
Train/test split -> standardize measurements -> Keras neural network
                                                    |
                                                    v
ESP32 serial output <- TensorFlow Lite Micro <- TFLite model as a C header
```

The input values are measurements in centimeters, in this order:

| Index | Measurement |
| --- | --- |
| 0 | Sepal length |
| 1 | Sepal width |
| 2 | Petal length |
| 3 | Petal width |

The output contains three probabilities:

| Class | Species |
| --- | --- |
| 0 | Iris setosa |
| 1 | Iris versicolor |
| 2 | Iris virginica |

Each input is standardized as `(measurement - mean) / scale`, using statistics calculated from the training split. The neural network has two dense layers of 16 ReLU units and a three-unit softmax output. It uses float32 inference; the conversion script does not quantize the model.

The included model occupies **3,804 bytes**, and the firmware reserves an **8 KiB tensor arena** for TensorFlow Lite Micro.

## Project files

```text
tinyml-iris-esp32/
├── include/iris_model.h    # Embedded trained model used by the firmware
├── scripts/train.py       # Training, TFLite conversion, and header export
├── src/main.cpp           # ESP32 initialization, inference, and serial output
├── platformio.ini         # ESP32 board, Arduino framework, and dependencies
├── diagram.json           # Wokwi ESP32 simulation diagram
├── wokwi.toml             # Wokwi firmware and ELF paths
├── requirements.txt       # Python training dependencies
└── .gitignore             # Excludes build output and local files
```

Training also produces a `models/` directory containing a Keras model, a TFLite model, and normalization parameters. These intermediate files are ignored by Git; the embedded model header is included because the firmware needs it.

## Build and run on an ESP32

You need an ESP32 development board compatible with PlatformIO's `esp32dev` target, a USB data cable, and PlatformIO Core or the PlatformIO extension for VS Code. The project uses the Arduino framework and the `tanakamasayuki/TensorFlowLite_ESP32` library, which PlatformIO installs during the build.

Clone the repository:

```bash
git clone https://github.com/ashkanchaji/tinyml-iris-esp32.git
cd tinyml-iris-esp32
```

If PlatformIO is not installed, you can install its CLI in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install platformio
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

Build, upload, and open the serial monitor:

```bash
pio run -e esp32dev
pio run -e esp32dev -t upload
pio device monitor -b 115200
```

If automatic port detection fails, use `pio device list`, then pass `--upload-port PORT` to the upload command or `--port PORT` to the monitor command. Exit the serial monitor with `Ctrl+C`.

The default sample in `src/main.cpp` is:

```cpp
float test_sample[] = {5.1, 3.5, 1.4, 0.2};
```

After successful initialization, serial output has this format:

```text
TFLite Micro setup completed successfully.
Predicted Class: <0, 1, or 2>
Probabilities - Class 0: <score>, Class 1: <score>, Class 2: <score>
```

To classify another flower, edit `test_sample`, rebuild, and upload. This demo uses a fixed sample; it does not read measurements from sensors or serial input.

## Run in Wokwi

The repository includes a Wokwi diagram for an ESP32 DevKit board and a configuration that points to PlatformIO's build output.

1. Open the project in VS Code with the Wokwi extension configured.
2. Build the firmware with `pio run -e esp32dev`.
3. Start the Wokwi simulator and inspect the serial output.

`wokwi.toml` expects these files:

```text
.pio/build/esp32dev/firmware.bin
.pio/build/esp32dev/firmware.elf
```

Rebuild before restarting the simulator after firmware changes. No external sensors are required.

## Retrain the model

Use a Python environment supported by your TensorFlow release. Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/train.py
```

The script:

1. Loads all 150 Iris samples from scikit-learn; no separate dataset download is needed.
2. Splits them into 120 training and 30 test samples with `random_state=42`.
3. Calculates the training mean and standard deviation and standardizes both splits.
4. Trains for 100 epochs with Adam, a batch size of 8, and sparse categorical cross-entropy.
5. Saves `models/iris_model.keras`, `models/iris_model.tflite`, and `models/iris_scaler_params.npz`.
6. Replaces `include/iris_model.h` with the newly converted model.

**Update the firmware normalization constants after retraining.** The script exports the model header but does not update the `mean` and `scale` arrays in `src/main.cpp`. Print the saved values with:

```bash
python -c "import numpy as np; p = np.load('models/iris_scaler_params.npz'); print('mean:', p['mean']); print('scale:', p['scale'])"
```

Copy those values into the corresponding arrays, then rebuild and upload. The constants currently in the firmware are rounded versions of the saved training statistics. If you change the data split or preprocessing, using the old constants will change the model's inputs.

## Notes and troubleshooting

- **Accuracy:** the training script reports validation accuracy, but the test split is also used as validation during training. This repository does not claim an independently measured test accuracy or hardware latency. Neural-network initialization is not seeded, so retraining can produce different weights and predictions.
- **`AllocateTensors() failed!`:** a replacement model may need more memory than the current 8 KiB arena. Check its memory needs and adjust `tensor_arena_size` before rebuilding.
- **`Model schema mismatch!`:** use a TensorFlow Lite model compatible with the TensorFlow Lite Micro library installed by PlatformIO.
- **Initialization errors:** the firmware skips inference if tensor setup fails. Resolve the reported schema or allocation error before retrying.
- **Dependencies:** the firmware uses Espressif 32 platform `7.0.1` and `TensorFlowLite_ESP32` library `1.0.0`. Python training packages are not version-pinned; converter updates or newly trained models may require compatibility adjustments.

This is a compact demonstration of training, model conversion, and on-device inference for an RTES/TinyML project.
