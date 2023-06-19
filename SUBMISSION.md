# **Setup**
1. Create a virtual environment with Python 3.9, preferably with `conda`. Use the following command:
```bash
conda create --name dive python=3.9	 
```
2. Activate the virtual environment:
```bash
conda activate dive
```
3. Installed the dependencies:
```bash
pip install -r requirements.txt
```
**Note:** In order to run all of the functionalities of the project successfully, it's necessary to have at lest 24 GB of RAM. It is also advisable to have a CUDA enabled GPU with at least 24 GB of VRAM in order to have a better experience when running the deep learning models.

# **Usage**
The project consist of a Python package named `dive`. Inside of this package there are three modules:
- `gen_data.py` which is in charge of all of the functionality related to generating the transcription for the meeting audio file. The main function of this module is `generate_transcription()` which generates a meeting transcription `pandas.DataFrame` that contains the desired information for the first objective. This `pandas.DataFrame` can then be easily exported to a CSV file.
- `identify_ais.py` which is in charge of identifying the action items given the meeting transcription in the expected format. the main function of this module is `identify_ais()` which generates a dictionary of the action items textual descriptions together with the assignee and the timestamps
- `utils.py` which contains general utility functions, for example to download a meeting video from YouTube.

## **Example**

```python 
from dive.utils import download_yt_video
from dive.gen_data import generate_transcription
from dive.identify_ais import identify_ais

# YouTuve url to a sample meeting video
yt_url = "https://www.youtube.com/watch?v=lBVtvOpU80Q"
audio_path = download_yt_video(yt_url)

# generate audio transcription with timestamps, speakers and text
transcription_df = generate_transcription(audio_path)
## the transcription can be easily exported to csv
# transcription_df.to_csv("transcription.csv", index=False)

# identify action items in the transcription dataframe
action_items = identify_ais(transcription_df)
```
Expected output:
```
[{
"text": " Create a list of key iterations aligned with Giddily cluster",
"assigne": " UNKNOWN",
"ts": "00:07:08",
},
{
"text": " Up-level the VS Code integrations",
"assigne": " Bria",
"ts": "00:10:00",
},
...,
{
"text": " Terraform integratio",
"assigne": " John Smit",
"ts": " 00:20:0"},
{
"text": ' "Pick the ones out of that list that apply to our stage.',
"assigne": " John Smit",
"ts": "00:25:11",
}]
```