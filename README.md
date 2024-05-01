# LLM-Vision-Datasets

In the era of large language models (LLMs), this repository is dedicated to collecting datasets, particularly focusing on image and video data for generative AI (such as diffusion models) and image-text paired data for multimodal models. We hope that the datasets shared by the community can help everyone better learn and train new AI models. By collaborating and sharing resources, we can accelerate the progress towards achieving Artificial General Intelligence (AGI). Let's work together to create a comprehensive collection of datasets that will serve as a valuable resource for researchers, developers, and enthusiasts alike, ultimately paving the way for the realization of AGI.

Some data is more suitable for generative AI, while other data is more suitable for VLM-related tasks. Some data is suitable for both.

If there is any infringement or inappropriate content, please contact us, and we will strive to prevent such content from appearing and handle it as soon as possible.

## Contents

- [Generative AI (Image Datasets)](#generative-ai-image)
  - [General image Datasets](#general-datasets-image)
  - [Datasets for Virtual Try-on](#datasets-for-virtual-try-on)
- [Generative AI (Video Datasets)](#generative-ai-video)
  - [General video Datasets](#general-datasets-video)
  - [Datasets of Pre-Training for Alignment](#datasets-of-pre-training-for-alignment)
  - [Datasets of Multimodal Instruction Tuning](#datasets-of-multimodal-instruction-tuning)
  
- [Multimodal Model Datasets](#multimodal-model-datasets)
- [Other Datasets](#other-datasets)
- [tools](#tools)
- [Awesome Datasets](#awesome-datasets)
- [How to Add a New Dataset](#how-to-add-a-new-dataset)

## Generative AI (Image)

High-resolution image data sharing, not limited to real-life or anime.

### General Datasets image

| Name  | Describe  | URL     |
|-|-|-|
| LAION | Released LAION-400M, LAION-5B, and other ultra-large image-text datasets, as well as various types of CLIP data. | https://laion.ai/projects/  <br>  https://huggingface.co/laion |
| Conceptual Captions Dataset |Conceptual Captions is a dataset containing (image-URL, caption) pairs designed for the training and evaluation of machine learned image captioning systems. | https://github.com/google-research-datasets/conceptual-captions  <br>  http://ai.google.com/research/ConceptualCaptions|
| laion-high-resolution-chinese|A subset from Laion5B-high-resolution (a multimodal dataset), around 2.66M image-text pairs (only Chinese).|https://huggingface.co/datasets/wanng/laion-high-resolution-chinese|

### Datasets for Virtual Try-on

| Name | Description | URL |
|------|-------------|------|
| StreetTryOn | A new in-the-wild virtual try-on dataset consisting of 12,364 and 2,089 street person images for training and validation. | [GitHub](https://github.com/cuiaiyu/street-tryon-benchmark) |
| CLOTH4D | A large-scale 4D dataset with 3D human, garment and texture models, SMPL pose parameters and HD images. | [GitHub](https://github.com/AemikaChow/CLOTH4D) |  
| DressCode | A dataset focused on modeling the underlying 3D geometry and appearance of a person and their garments given a few or a single image. | [Download Form](https://docs.google.com/forms/d/e/1FAIpQLSeWVzxWcj3JSALtthuw-2QDAbf2ymiK37sA4pRQD4tZz2vqsw/viewform), [Paper](https://arxiv.org/pdf/2204.08532.pdf) |
| VITON-HD | A high-resolution virtual try-on dataset with 13,679 image pairs at 1024 x 768 resolution. | [Download](https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0), [Project Page](https://psh01087.github.io/VITON-HD/) |
| VITON | The first image-based virtual try-on dataset with 16,253 image pairs. | [Download](https://drive.google.com/file/d/1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo/view), [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Han_VITON_An_Image-Based_CVPR_2018_paper.pdf) |
| MPV | Multi-Pose Virtual try-on dataset containing 35,687/13,524 person/clothing images. | [Download](https://drive.google.com/drive/folders/1e3ThRpSj8j9PaCUw8IrqzKPDVJK_grcA), [Paper](https://arxiv.org/abs/1902.11026) |
| Deep Fashion3D | A large-scale 3D garment dataset with diverse garment styles and rich annotations. | [Paper](https://arxiv.org/abs/2003.12753) |
| DeepFashion MultiModal | A dataset for multi-modal virtual try-on containing unpaired people and garment images. | [Download](https://github.com/yumingj/DeepFashion-MultiModal) |
| Digital Wardrobe | High-quality 3D garments from real consumer photos with 2D-3D alignment annotations. | [Download/Paper/Project](http://virtualhumans.mpi-inf.mpg.de/mgn/) |  
| TailorNet Dataset | Paired images of clothed 3D humans with consistent geometry and pose for garment transfer. | [Download](https://github.com/zycliao/TailorNet_dataset), [Project](http://virtualhumans.mpi-inf.mpg.de/tailornet/) |
| CLOTH3D | The first 3D garment dataset with digital garments and 3D human models. | [Paper](https://arxiv.org/abs/1912.02792) |
| 3DPeople | 3D human dataset with 80 subjects wearing diverse clothing and poses. | [Project](https://www.albertpumarola.com/research/3DPeople/index.html) | 
| THUman Dataset | High-resolution 3D textured human body dataset with 7000+ models of 200+ subjects. | [Project](http://www.liuyebin.com/deephuman/deephuman.html) |
| Garment Dataset | 3D garment dataset with digital garments fitted to real people and garment images. | [Project](http://geometry.cs.ucl.ac.uk/projects/2018/garment_design/) |

## Generative AI (Video)

High-resolution video data sharing, not limited to real-life or anime.

### General Datasets (video)

| Name  | Describe  | URL     |
|-|-|-|

## Multimodal Model Datasets

This section includes datasets for multimodal models,requiring at least images and corresponding captions, suitable for training multimodal large models.

### Datasets of Pre-Training for Alignment

| Name  | Describe  | URL     |
|-|-|-|
| LAION | Released LAION-400M, LAION-5B, and other ultra-large image-text datasets, as well as various types of CLIP data. | https://laion.ai/projects/  <br>  https://huggingface.co/laion |
| Conceptual Captions Dataset |Conceptual Captions is a dataset containing (image-URL, caption) pairs designed for the training and evaluation of machine learned image captioning systems. | https://github.com/google-research-datasets/conceptual-captions  <br>  http://ai.google.com/research/ConceptualCaptions|
| COYO-700M | COYO-700M: Large-scale Image-Text Pair Dataset for training and evaluation of machine learning-based image-text matching models. | https://github.com/kakaobrain/coyo-dataset/ |  
| ShareGPT4V | ShareGPT4V: A large image-text dataset with captions generated by GPT-4 to improve multi-modal models. | https://arxiv.org/pdf/2311.12793.pdf |
| AS-1B | The All-Seeing Project dataset with over 1 billion regions annotated with semantic tags, QA pairs, and captions, for panoptic visual recognition. | https://arxiv.org/pdf/2308.01907.pdf |
| InternVid | InternVid: A large-scale video-text dataset for multimodal understanding and generation. | https://arxiv.org/pdf/2307.06942.pdf |
| MS-COCO | Microsoft COCO: A large-scale object detection, segmentation, and captioning dataset. | https://arxiv.org/pdf/1405.0312.pdf |
| SBU Captions | SBU Captioned Photo Dataset containing 1 million images with user-associated captions harvested from Flickr. | https://proceedings.neurips.cc/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf |
| Conceptual Captions | Conceptual Captions: A cleaned, web-scraped image alt-text dataset for training image captioning models. | https://aclanthology.org/P18-1238.pdf |
| LAION-400M | LAION-400M: An open, large-scale dataset of 400 million CLIP-filtered image-text pairs. | https://arxiv.org/pdf/2111.02114.pdf <br> https://laion.ai/projects/ <br> https://huggingface.co/laion |
| VG Captions | Visual Genome dataset connecting structured image concepts to language with crowdsourced annotations. | https://link.springer.com/content/pdf/10.1007/s11263-016-0981-7.pdf |
| Flickr30k | Flickr30k Entities: A dataset of 30k images with 5 captions each, annotated with bounding boxes and entity mentions. | https://openaccess.thecvf.com/content_iccv_2015/papers/Plummer_Flickr30k_Entities_Collecting_ICCV_2015_paper.pdf |  
| AI-Caps | AI Challenger: A large-scale Chinese dataset with millions of images and natural language descriptions. | https://arxiv.org/pdf/1711.06475.pdf |
| Wukong Captions | Wukong: A 100 million scale Chinese cross-modal pre-training benchmark dataset. | https://proceedings.neurips.cc/paper_files/paper/2022/file/a90b9a09a6ee43d6631cf42e225d73b4-Paper-Datasets_and_Benchmarks.pdf |
| GRIT | Grounded Multimodal Language Model dataset containing images aligned with text segments. | https://arxiv.org/pdf/2306.14824.pdf |
| Youku-mPLUG | Youku-mPLUG: A 10 million scale Chinese video-language pre-training dataset. | https://arxiv.org/pdf/2306.04362.pdf |
| MSR-VTT | MSR-VTT: A large video description dataset for bridging video and language. | https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf |
| Webvid10M | Webvid-10M: A large-scale video-text dataset for joint video-language representation learning. | https://arxiv.org/pdf/2104.00650.pdf |
| WavCaps | WavCaps: A ChatGPT-assisted weakly-labeled audio captioning dataset. | https://arxiv.org/pdf/2303.17395.pdf |
| AISHELL-1 | AISHELL-1: An open-source Mandarin speech corpus and a speech recognition baseline. | https://arxiv.org/pdf/1709.05522.pdf |  
| AISHELL-2 | AISHELL-2: An industrial-scale Mandarin speech recognition dataset. | https://arxiv.org/pdf/1808.10583.pdf |
| VSDial-CN | Visual Semantic Dialog in Chinese: an image-audio-text dataset for studying multimodal language models. | https://arxiv.org/pdf/2305.04160.pdf |


### Datasets of Multimodal Instruction Tuning

| Name | Describe | URL |
|:-----|:---------|:----|
| CogVLM-SFT-311K | CogVLM-SFT-311K, a crucial aligned corpus for initializing CogVLM v1.0, was constructed by first selecting approximately 3,500 high-quality data samples from the open-source MiniGPT-4 (minigpt4-3500). This subset was then combined with Llava-Instruct-150K and machine-translated into Chinese. | https://github.com/THUDM/CogVLM/blob/main/dataset.md |
| **ALLaVA-4V** | Multimodal instruction dataset generated by GPT4V | https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V |
| **IDK** | Dehallucinative visual instruction for "I Know" hallucination | https://github.com/ncsoft/idk |
| **CAP2QA** | Image-aligned visual instruction dataset | https://github.com/ncsoft/cap2qa |
| **M3DBench** | A large-scale 3D instruction tuning dataset | https://github.com/OpenM3D/M3DBench |
| **ViP-LLaVA-Instruct** | A mixture of LLaVA-1.5 instruction data and the region-level visual prompting data | https://huggingface.co/datasets/mucai/ViP-LLaVA-Instruct |
| **LVIS-Instruct4V** | A visual instruction dataset via self-instruction from GPT-4V | https://huggingface.co/datasets/X2FD/LVIS-Instruct4V |
| **ComVint** | A synthetic instruction dataset for complex visual reasoning | https://github.com/RUCAIBox/ComVint#comvint-data |
| **SparklesDialogue** | A machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions to augment the conversational competence of instruction-following LLMs across multiple images and dialogue turns. | https://github.com/HYPJUDY/Sparkles#sparklesdialogue |
| **StableLLaVA** | A cheap and effective approach to collect visual instruction tuning data | https://github.com/icoz69/StableLLAVA |
| **M-HalDetect** | A dataset used to train and benchmark models for hallucination detection and prevention | [Coming soon]() |
| **MGVLID** | A high-quality instruction-tuning dataset including image-text and region-text pairs | - |
| BuboGPT | BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs | https://huggingface.co/datasets/magicr/BuboGPT |
| SVIT | SVIT: Scaling up Visual Instruction Tuning | https://huggingface.co/datasets/BAAI/SVIT |
| mPLUG-DocOwl | mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding | https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocLLM |
| PF-1M | Visual Instruction Tuning with Polite Flamingo | https://huggingface.co/datasets/chendelong/PF-1M/tree/main |
| ChartLlama | ChartLlama: A Multimodal LLM for Chart Understanding and Generation | https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset |
| LLaVAR | LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding | https://llavar.github.io/#data |
| MotionGPT | MotionGPT: Human Motion as a Foreign Language | https://github.com/OpenMotionLab/MotionGPT |
| LRV-Instruction | Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning | https://github.com/FuxiaoLiu/LRV-Instruction#visual-instruction-data-lrv-instruction |
| Macaw-LLM | Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration | https://github.com/lyuchenyang/Macaw-LLM/tree/main/data |
| LAMM-Dataset | LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark | https://github.com/OpenLAMM/LAMM#lamm-dataset |
| Video-ChatGPT | Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models | https://github.com/mbzuai-oryx/Video-ChatGPT#video-instruction-dataset-open_file_folder |
| MIMIC-IT | MIMIC-IT: Multi-Modal In-Context Instruction Tuning | https://github.com/Luodian/Otter/blob/main/mimic-it/README.md |
| M³IT | M³IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning | https://huggingface.co/datasets/MMInstruction/M3IT |
| LLaVA-Med | LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day | [Coming soon](https://github.com/microsoft/LLaVA-Med#llava-med-dataset) |
| GPT4Tools | GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction | [Link](https://github.com/StevenGrove/GPT4Tools#dataset) |
| MULTIS | ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst | [Coming soon](https://iva-chatbridge.github.io/) |
| DetGPT | DetGPT: Detect What You Need via Reasoning | [Link](https://github.com/OptimalScale/DetGPT/tree/main/dataset) |
| PMC-VQA | PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering | [Coming soon](https://xiaoman-zhang.github.io/PMC-VQA/) |
| VideoChat | VideoChat: Chat-Centric Video Understanding | [Link](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) | 
| X-LLM | X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages | [Link](https://github.com/phellonchen/X-LLM) |
| LMEye | LMEye: An Interactive Perception Network for Large Language Models | [Link](https://huggingface.co/datasets/YunxinLi/Multimodal_Insturction_Data_V2) |
| cc-sbu-align | MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models | [Link](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) |
| LLaVA-Instruct-150K | Visual Instruction Tuning | [Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| MultiInstruct | MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning | [Link](https://github.com/VT-NLP/MultiInstruct) |



## Other Datasets

Other datasets that are not easily categorized at the moment.

| Name  | Describe  | URL     |
|-|-|-|

## Tools

Tools that aid in data acquisition and cleaning.

| Name  | Describe  | URL     |
|-|-|-|
|img2dataset |Easily turn large sets of image urls to an image dataset. Can download, resize and package 100M urls in 20h on one machine. | https://github.com/rom1504/img2dataset|
| | | | 


## Awesome datasets 

Thanks to the help from the following Awesome repositories.

- https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models

- https://github.com/Atomic-man007/Awesome_Multimodel_LLM

- https://github.com/minar09/awesome-virtual-try-on

## How to Add a New Dataset

To contribute a new dataset to this repository, please follow these steps:

1. Fork this repository to your own GitHub account.
2. Add your dataset to the appropriate section in the README file, following the format:

```
| Name  | Describe  | URL     |
|-|-|-|
|DatasetsName | DatasetsDescribe | DatasetsURL1 <br> DatasetsURL2 <br> DatasetsURL3 |
```

For example:
```
| Name  | Describe  | URL     |
|-|-|-|
| LAION | Released LAION-400M, LAION-5B, and other ultra-large image-text datasets, as well as various types of CLIP data. | https://laion.ai/projects/  <br>  https://huggingface.co/laion |
```
If your dataset doesn't fit into any of the existing categories, create a new section for it in the README file.

3. Commit and push, Create a pull request.

By following these steps, you can help expand the collection of datasets available in this repository and contribute to the advancement of generative AI and multimodal visual AI research.

Contributions of more datasets related to generative AI and multimodal visual AI are welcome! If you have any suggestions or comments, please feel free to open an issue.


