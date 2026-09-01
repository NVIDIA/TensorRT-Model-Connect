# Sample Node.js App using TRT-Model-Connect

This is a demonstration application showing how to integrate the TensorRT-Model-Connect Node.js bindings into a real-world web application using Express.js.

## Prerequisites
- Node.js >= 18.0.0
- The local `tensorrt-model-connect-node` binding must be built.

## Setup
1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the application:
   ```bash
   npm start
   ```

## Usage
Send a POST request to the API:

```bash
curl -X POST http://localhost:3000/api/generate \
-H "Content-Type: application/json" \
-d '{"prompt": "Hello World!", "max_new_tokens": 100}'
```

You should receive a JSON response containing the simulated AI text and inference metrics.
