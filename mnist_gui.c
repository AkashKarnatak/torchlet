#include "raylib.h"
#include "tensor.h"
#include <stdint.h>
#include <stdio.h>

#define SCREEN_WIDTH 620
#define SCREEN_HEIGHT 760
#define CANVAS_SIZE 600
#define BRUSH_SIZE 60
#define GRID_SIZE 28

void sgd_step(Tensor *tensor, Tensor *grad, float lr) {
  size_t numel;

  numel = tensor_numel(tensor);

  for (size_t i = 0; i < numel; ++i) {
    tensor->data[i] -= lr * grad->data[i];
  }
}

int32_t OnDrawingFinished(RenderTexture2D canvas, Tensor *img, Tensor **params,
                          size_t n_params) {
  Tensor *w1, *b1, *w2, *b2;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6, *out7;
  Image image;
  Color *pixels;
  int32_t guess;

  image = LoadImageFromTexture(canvas.texture);
  ImageResize(&image, GRID_SIZE, GRID_SIZE);
  ImageFlipVertical(&image);

  pixels = LoadImageColors(image);
  for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
    img->data[i] = 1.0f - (pixels[i].r / 255.0f);
  }

  w1 = params[0], b1 = params[1], w2 = params[2], b2 = params[3];

  out1 = tensor_matmul(img, w1);
  out2 = tensor_matadd(out1, b1);
  out3 = tensor_relu(out2);
  out4 = tensor_matmul(out3, w2);
  out5 = tensor_matadd(out4, b2);
  out6 = tensor_softmax(out5, -1);
  out7 = tensor_argmax_at(out6, -1);

  tensor_print(out6);
  guess = out7->data[0];

  tensor_free(out1);
  tensor_free(out2);
  tensor_free(out3);
  tensor_free(out4);
  tensor_free(out5);
  tensor_free(out6);
  tensor_free(out7);

  UnloadImageColors(pixels);
  UnloadImage(image);

  return guess;
}

int main() {
  const size_t n_params = 4;
  Tensor *params[n_params];
  Tensor *img;

  // load mnist model
  load_model("model.bin", params, n_params);
  img = tensor_empty((size_t[]){1, 784}, 2);
  tensor_debug(img);

  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Digit Drawer");
  RenderTexture2D canvas = LoadRenderTexture(CANVAS_SIZE, CANVAS_SIZE);
  SetTargetFPS(60);

  // Clear canvas initially
  BeginTextureMode(canvas);
  ClearBackground(WHITE);
  EndTextureMode();

  Vector2 lastMousePos = {-1, -1};
  Rectangle clearButton = {10, CANVAS_SIZE + 20, 100, 30};
  bool wasDrawing = false;
  int32_t guessedNumber = -1;

  while (!WindowShouldClose()) {
    Vector2 mousePos = GetMousePosition();
    bool isDrawing = IsMouseButtonDown(MOUSE_LEFT_BUTTON);

    // Draw on the canvas
    if (isDrawing) {
      if (mousePos.x < CANVAS_SIZE && mousePos.y < CANVAS_SIZE) {
        BeginTextureMode(canvas);
        if (lastMousePos.x != -1 && lastMousePos.y != -1) {
          DrawLineEx(lastMousePos, mousePos, BRUSH_SIZE, BLACK);
        }
        DrawCircle(mousePos.x, mousePos.y, BRUSH_SIZE / 2.0f, BLACK);
        EndTextureMode();
        lastMousePos = mousePos;
        wasDrawing = true;
      }
    } else {
      if (wasDrawing) {
        guessedNumber = OnDrawingFinished(canvas, img, params, n_params);
        wasDrawing = false;
      }
      lastMousePos.x = -1;
      lastMousePos.y = -1;
    }

    // Clear canvas
    if (IsKeyPressed(KEY_C) ||
        (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) &&
         CheckCollisionPointRec(mousePos, clearButton))) {
      BeginTextureMode(canvas);
      ClearBackground(WHITE);
      EndTextureMode();
      guessedNumber = -1;
    }

    BeginDrawing();
    ClearBackground(LIGHTGRAY);

    // Draw the canvas
    DrawTextureRec(canvas.texture, (Rectangle){0, 0, CANVAS_SIZE, -CANVAS_SIZE},
                   (Vector2){10, 10}, WHITE);

    DrawRectangleRec(clearButton, DARKGRAY);
    DrawText("Clear", (int)(clearButton.x + 25), (int)(clearButton.y + 7), 20,
             WHITE);

    DrawText("Draw a digit (0-9)", 10, CANVAS_SIZE + 60, 20, DARKGRAY);
    DrawText("Press 'C' to Clear", 10, CANVAS_SIZE + 85, 20, DARKGRAY);

    if (guessedNumber > -1) {
      char guessText[20];
      sprintf(guessText, "Guessed: %d", guessedNumber);
      DrawText(guessText, 10, CANVAS_SIZE + 120, 30, BLACK); // Larger font size
    }

    EndDrawing();
  }

  UnloadRenderTexture(canvas);
  CloseWindow();

  for (size_t i = 0; i < n_params; ++i) {
    tensor_free(params[i]);
  }
  tensor_free(img);

  return 0;
}
