# **Approach**

[**Loom Demo Video**](https://www.loom.com/share/074c1c1624d842f1b75662ca86a9c7a7?sid=adda0c3e-84c1-4b05-8f65-04071f800eba)

### **Data generation**

For the data generation objective I considered mainly two options, and explored both of them:
- Look for meetings dataset on the internet, especially on Kaggle or HuggingFace.
	For this option I found the [AMI meeting corpus](https://groups.inf.ed.ac.uk/ami/corpus/) dataset from 2006. It comprises over 100 meetings, predominantly centered around a product design team. The dataset offers multiple annotations, such as transcribed text, speaker identification, and additional information. One particularly intriguing aspect is the inclusion of dialogue act annotations, which bear a resemblance to action items.
- Generate the data from scratch using publicly available meeting videos on YouTube. Which is the option that I dedicated the most time into.
	For this option I thought of automatically generating the data instead of doing it manually. To achieve this I decomposed the task in smaller ones:
	- First, Transcribe the audio files. I have some previous experience using the Whisper ASR model from OpenAI, so I immediately went for this one. 
		- I choose the "small.en" model assuming that the meetings that we are interested in are mostly in English and went for the version `small.en` as it provided a slightly better performance that the base one, and didn't take much longer to transcribe
		- I also went from the implementation from the library [`faster_whisper`](https://github.com/pyannote/pyannote-audio) which promises to be 4 times faster that OpenAIs implementation while maintaining the same performance.
		-  Whisper also outputs the timestamps so that's also a plus.
	- Second, diarize the audio in order to identify when speakers are talking. For this I used the `PretrainedSpeakerEmbedding` model from the [`pyanote.audio`](https://github.com/pyannote/pyannote-audio) library to generate embeddings for each segment of the audio and then clustered them together to get the final transcription. I took most of the implementation from [this](https://huggingface.co/spaces/vumichien/Whisper_speaker_diarization) HF space.
	I found that the results were quite good but not perfect as is to expect.	I decided to go with this approach as I find it to be more useful for an actual real product situation and also it able me to practice and explore some interesting tools, which is always fun.
  
## **Action Item Identification**

This is the task I found most difficult. At first I didn't had a clear idea of how to model this problem as a common NLP task. I first tough about framing it as a NER task, where each action item is a specific token or as a text classification where I took every sentence and classified them one by one between action item or not. 

But there was a big problem with these approaches and it was that a didn't had an annotated dataset for action items. So instead I tough of using LLMs. This seemed like a rather straight forward solution to me and I was confident it wouldn't give horrible results because I have some previous experience working with [LLaMA](https://github.com/facebookresearch/llama) and more specifically with the [Alpaca-Lora](https://github.com/tloen/alpaca-lora) model. Now, before I explain how I went about this, there's some considerations that are worth noting.
- First, LLMs are expensive to run. Much more expensive compared to the typical Bert based models. But also not too expensive as the LLaMA 7B model can run on a Kaggle instance with 2 T4 Nvidia GPUs or, in general, any GPU with at least 24 GB of VRAM.
- Second, LLMs are hard to control. Finding the right prompt is always a challenge and even then, they can always hallucinate at times. On the other hand, LLaMA 7B and specifically Alpaca-Lora, which was trained on an instruction dataset, I thought were decently suitable for this task.
Now onto the actual method. First I chose the [Alpaca-Lora](https://huggingface.co/tloen/alpaca-lora-7b) model from HuggingFace, because as I already said, I have some previous experience with it and gave me good results. One important consideration is that as this model is based on LLaMA, and for that reason it also holds their restrictive license, which means that is not suitable for commercial use. An alternative to this would be to use a similar model like the recent [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b-instruct), specifically their instruct version.

Next was prompt engineering. Firstly, I cannot feed the entire transcription of the meeting to the LLM as it clearly exceeds its context length for longer meetings. Instead I figured a couple of dialogues would give enough context to identify an action item present inside of them. Also I gave all the information, meaning not just the text but also the speaker and the timestamp.
 
When creating the prompt, I found that it worked best if I split the action item identification into smaller tasks. Namely, first identify if there are any action items present in the dialogue excerpt, if the answer is positive I asked the model to literally give me the action item, together with the assignee and the timestamp. To accomplish this i used the [`guidance`](https://github.com/microsoft/guidance) library, which provided a template like DSL that can "interleave generation, prompting, and logical control into a single continuous flow matching how the language model actually processes the text".

Finally, the results were good but not too good. Mainly because there is quite a lot of false positives and lacks in correctness, for instance mentioning a speaker that was not actually participating in the conversation. On the other hand, there are some possible reasons for this:
- First, action items are often ambiguous and difficult to identify. They also are often times not self contained in a single sentence but spread out in the context of multiple dialogues
- Second, action items are extremely sparse, meaning that in a 1 hour meeting there likely going to be a few action items but close to thousand of sentences.

At this point I tried to consult the state of the art in order to look for better alternative solutions. As a side note, it's worth nothing that the current work on action item identification is really minimal. I couldn't find any information on the internet or models/datasets on HuggingFace. All I found were some companies that offered the task as a service(like dive :)). With concordance with this, the literature was also scarce but I manage to find two relevant papers.
- [MEETING ACTION ITEM DETECTION WITH REGULARIZED CONTEXT MODELING](https://arxiv.org/pdf/2303.16763.pdf) 2023 paper for Alibaba. This paper tries to tackle the task by approaching it as binary sentence classification problem. They value they provide is that they implement a technique to use the context information in a nuance way. Tough it has a lot of problems.
	- Code is hardly reproducible.
	- Code structure is bad.
	- Code readability is bad.
	- Outdated versions(TensorFlow 1.\*)
	- The use the AMI dataset for this tasks. With some questionable indirect annotations as they use the action dialogues as if they were action item annotations.
- [Automatic Rephrasing of Transcripts-based Action Items](https://www.microsoft.com/en-us/research/publication/automatic-rephrasing-of-transcripts-based-action-items/) 2021 paper from Microsoft. The important idea is that they understand that action items are often not expressed in a single sentence but span multiple sentence in a variety of ways. They aim to condense the information regarding action items that is all over the dialogue of the meeting participants, into a single self-contained sentence.

Out of the two, the second one was the one that looked more promising but after contacting with the authors I found the neither the dataset nor the source code were publicly released. 

In conclusion, I think that it is possible to improve the results guided by some of the ideas presented in those papers, mainly fine tuning on a custom dataset, but perhaps the biggest limitation is the lack of an annotated meeting action item dataset. One could think of building one from scratch, but it is likely going to take some a lot of time and effort, even with help of LLMs.

It was a fun challenge, perhaps this documents was much longer that what it needed to be :P.
