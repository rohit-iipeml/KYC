from controllers.kyc_controller import *
import uvicorn

# Run the server locally
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)