# **Design Genie: AI-Powered Automated Poster Generation Tool**

Design Genie is an innovative AI-driven tool that automates the creation of personalized event posters. Built using **Stable Diffusion**, **NLP techniques**, and **deep learning models**, the tool combines textual and visual inputs to generate high-quality, contextually relevant posters with enhanced slogans and dynamic scene generation. It is optimized for resource efficiency, minimizing GPU usage and CO₂ emissions.

---

## **Features**
1. **Prompt Enhancement Using NLP**:
   - Analyzes user inputs using techniques like POS tagging, Named Entity Recognition (NER), and sentiment analysis (VADER and DistilBERT).
   - Dynamically refines prompts to create contextually rich inputs for the Stable Diffusion model.

2. **Dual Input System**:
   - Combines textual prompts with relevant image inputs to improve thematic accuracy and generate visually cohesive posters.

3. **Dynamic Scene Generation**:
   - Automatically adapts color palettes, time of day, and weather conditions based on sentiment analysis for mood-aligned designs.

4. **Slogan Generation and Integration**:
   - Generates catchy slogans using the **Gemini 1.5 Flux model** and seamlessly integrates them into the poster using object detection and free space analysis.

5. **Resource Efficiency**:
   - Reduces GPU utilization and carbon footprint with optimized prompt processing and enhanced image generation techniques.

---

## **System Architecture**
The system is composed of the following modules:
- **Prompt Enhancement**: Refines user inputs with NLP.
- **Dynamic Scene Generation**: Generates attributes and settings based on sentiment and user intent.
- **Image Matching and Input**: Merges user prompts with relevant images.
- **Stable Diffusion**: Generates high-quality posters based on enriched prompts and images.
- **Slogan Integration**: Detects free space and places generated slogans into the poster.

Refer to the `docs/System_Architecture.pdf` for a detailed diagram of the architecture.

---

## **Technologies Used**
- **Stable Diffusion**: For high-quality image generation.
- **NLP Libraries**: 
  - `spaCy` for entity recognition and tokenization.
  - `VADER` for lexicon-based sentiment analysis.
  - `DistilBERT` for contextual sentiment analysis.
- **Image Processing**: `PyTorch`, `OpenCV`.
- **Slogan Generation**: Gemini 1.5 Flux model.
- **Performance Monitoring**: `CodeCarbon`, `Weights & Biases`.

---

## **Installation and Setup**
### Prerequisites
1. Python 3.8+ installed.
2. GPU-enabled system for running Stable Diffusion (optional but recommended).
3. Required Python packages:
   - `transformers`
   - `torch`
   - `spacy`
   - `opencv-python`
   - `codecarbon`
   - `wandb`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/design-genie.git
   cd design-genie
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pre-trained models for Stable Diffusion and place them in the `models/` directory.
4. Set up the `.env` file with required API keys (e.g., Gemini API for slogan generation).
5. Run the application:
   ```bash
   python app.py
   ```

---

## **Usage**
1. **Input a Text Prompt**: Describe your event (e.g., "A vibrant music festival with neon lights").
2. **Optional: Provide an Image**: Add a supporting image for better context.
3. **Generate the Poster**: The system processes the input and outputs a high-quality event poster with an integrated slogan.

---

## **Performance and Metrics**
- **Efficiency**:  
   - Reduced GPU usage by 20% compared to baseline models.
   - Lower CO₂ emissions (measured via CodeCarbon).
- **Quality**:  
   - Enhanced posters scored higher in clarity, creativity, and alignment during user evaluations.


## **Future Scope**
- Add multilingual support for diverse audiences.
- Real-time customization via a web interface.
- Expand to other creative domains, such as banners and brochures.

---

## **Contributing**
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request. Refer to the `CONTRIBUTING.md` for detailed guidelines.

---

## **Acknowledgments**
- Guided by **Dr. Abirami Murugappan**, Anna University.
- Developed by **Shantha Priya M**, **Nithya Sree K**, **Sandhya Shankar**, and **Sabari Srinath R**.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.
