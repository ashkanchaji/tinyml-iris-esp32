#include <Arduino.h>
#include "iris_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Global variables for TFLite Micro
const int tensor_arena_size = 8 * 1024;
uint8_t tensor_arena[tensor_arena_size];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Normalization parameters from Python (Replace these with your actual output values)
float mean[] = {5.8092, 3.0617, 3.7267, 1.1833};
float scale[] = {0.8204, 0.4472, 1.745, 0.7491};

// Sample test data (SepalLength, SepalWidth, PetalLength, PetalWidth)
float test_sample[] = {5.1, 3.5, 1.4, 0.2};

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Load the model from header file
  model = tflite::GetModel(iris_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  // Load all operations
  static tflite::AllOpsResolver resolver;

  // Build the interpreter
  static tflite::MicroErrorReporter error_reporter;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, tensor_arena_size, &error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }

  // Get input and output tensors pointers
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLite Micro setup completed successfully.");
}

void loop() {
  if (input == nullptr || output == nullptr) {
    delay(5000);
    return;
  }

  // Normalize the input data using mean and scale
  for (int i = 0; i < 4; i++) {
    input->data.f[i] = (test_sample[i] - mean[i]) / scale[i];
  }

  // Run model inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  // Read probability outputs
  float class_0_prob = output->data.f[0];
  float class_1_prob = output->data.f[1];
  float class_2_prob = output->data.f[2];

  // Find the class with highest probability
  int predicted_class = 0;
  float max_prob = class_0_prob;
  if (class_1_prob > max_prob) {
    max_prob = class_1_prob;
    predicted_class = 1;
  }
  if (class_2_prob > max_prob) {
    max_prob = class_2_prob;
    predicted_class = 2;
  }

  // Print predicted class and scores
  Serial.print("Predicted Class: ");
  Serial.println(predicted_class);
  Serial.print("Probabilities - Class 0: ");
  Serial.print(class_0_prob);
  Serial.print(", Class 1: ");
  Serial.print(class_1_prob);
  Serial.print(", Class 2: ");
  Serial.println(class_2_prob);

  delay(5000);
}
