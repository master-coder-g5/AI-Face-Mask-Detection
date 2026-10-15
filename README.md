\# AI Face Mask Detection



An AI-based face mask detection system using Python, PyTorch, MobileNetV2, and OpenCV. The system uses a webcam to detect a person's mask category and can send the prediction and confidence score to a Node-RED dashboard over TCP.



\## Features



\- Real-time webcam-based face mask classification

\- Four prediction classes: CLOTH, N95, NO\_MASK, and SURGICAL

\- Deep learning model trained using PyTorch and MobileNetV2

\- Prediction confidence displayed as a percentage

\- TCP communication with Node-RED using JSON messages

\- Designed for possible Arduino UNO R4 hardware integration



\## Detection Classes



| Class | Description |

|---|---|

| `CLOTH` | Cloth face covering |

| `N95` | N95 mask |

| `NO\_MASK` | No mask detected |

| `SURGICAL` | Surgical face mask |



\## Technologies Used



\- Python

\- PyTorch and Torchvision

\- OpenCV

\- MobileNetV2

\- Node-RED

\- Arduino UNO R4 (hardware integration)



\## Project Structure



```text

AI-Face-Mask-Detection/

├── camera\_test.py

├── check\_dataset.py

├── collect\_cloth.py

├── collect\_images.py

├── copy\_dataset.py

├── face\_detection.py

├── face\_test.py

├── mask\_type\_model.pth

├── split\_dataset.py

├── test\_mask\_webcam.py

├── train\_model.py

└── .gitignore

```



The trained model is included in this repository. The virtual environment and image datasets are excluded.



\## Requirements



\- Python 3.12 or a compatible Python version

\- A working webcam

\- Git

\- Node.js and Node-RED for dashboard integration



\## Installation



Clone this repository:



```bash

git clone https://github.com/master-coder-g5/AI-Face-Mask-Detection.git

cd AI-Face-Mask-Detection

```



Create and activate a virtual environment on Windows:



```cmd

python -m venv venv

venv\\Scripts\\activate

```



Install the required Python libraries:



```cmd

python -m pip install --upgrade pip

pip install torch torchvision opencv-python pillow numpy

```



\## Run the Detection System



Make sure the trained model `mask\_type\_model.pth` is present in the project directory.



Start the webcam detection program:



```cmd

python test\_mask\_webcam.py

```



Allow camera access if prompted. Press \*\*Q\*\* to quit the webcam window.



\## Node-RED Integration



The detection program can send predictions to Node-RED using TCP on port `5005`, provided the Node-RED flow is configured to listen on that port.



Example JSON message:



```json

{

&#x20; "mask": "CLOTH",

&#x20; "confidence": 92.4

}

```



The Node-RED flow can use these values to update dashboard widgets:



\- `msg.payload.mask` — predicted mask category

\- `msg.payload.confidence` — prediction confidence percentage



Configure the TCP input and message parsing to match the format sent by the Python program.



\## Model Performance



The saved model recorded approximately \*\*95.05% validation accuracy\*\* during training. Actual webcam predictions may vary depending on lighting, camera quality, face position, and mask appearance.



\## Dataset and Training



The image datasets are not included in this repository. To train the model again, prepare a compatible dataset and follow the folder structure and preprocessing expected by the training scripts.



\## Hardware Integration



The project is intended to support integration with an Arduino UNO R4. Hardware communication and indicator behavior must be configured and tested separately.



\## Author



\*\*GitHub:\*\* \[master-coder-g5](https://github.com/master-coder-g5)



\## License



No license has been added yet. Please contact the repository owner before reusing or redistributing this project.



