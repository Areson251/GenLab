import os
import numpy as np
import torch
from PIL import Image
import gradio as gr
from .stable_diffusion import StableDiffusionModel
from .kandinsky import KandinskyModel
import logging
import time 

class GradioWindow():
    def __init__(self) -> None:
        self.path_to_orig_imgs = None
        self.path_to_output_imgs = None
        self.path_to_prompts = None
        self.path_to_negative_prompts = None
        self.path_to_logs = None

        self.original_img = None
        self.masks = None
        self.prompts = None
        self.logger = None
        self.stable_diffusion = None 
        self.kandinsky = None 

        #TODO: rewrite time calculating to something goodx
        self.sd_avg_time = 0
        self.kandinsky_avg_time = 0

        self.folders = [self.path_to_orig_imgs, self.path_to_output_imgs]

        # self.check_folders()
        # self.start_logging()
        # self.read_prompts()
        # self.load_models()

        self.main()

    def start_logging(self):
        self.logger = logging.getLogger(__name__)
        path = os.path.join(self.path_to_output_imgs, self.path_to_logs)
        logging.basicConfig(filename=path, level=logging.INFO)
        self.logger.info("Started")

    def check_folders(self):
        for path in self.folders:
            if not os.path.exists(path):
                os.makedirs(path)

    def read_prompts(self):
        with open(self.path_to_prompts, "r") as file:
            self.prompts = [line.rstrip() for line in file]
            self.logger.info("Use prompts from "+self.path_to_prompts)

    def load_models(self):
        self.stable_diffusion = StableDiffusionModel()
        # self.kandinsky = KandinskyModel()
        self.logger.info("Models loaded")

    def main(self):
        with gr.Blocks() as self.demo:
            with gr.Row():
                self.input_img = gr.ImageEditor(
                    type="pil",
                    label="Input",
                )
                with gr.Column():
                    self.path_to_orig_imgs = gr.Textbox(label="Images path", value="images/orig_imgs")
                    self.path_to_output_imgs = gr.Textbox(label="Output path", value="images/output_imgs/mem")
                    self.path_to_prompts = gr.Textbox(label="Prompt path", value="prompts/mem.txt")
                    self.path_to_negative_prompts = gr.Textbox(label="Negative prompt path", value="images/negative_prompts.txt")
                    self.path_to_logs = gr.Textbox(label="Logs file name", value="out.log")
                    self.setup_settings = gr.Button("Set up")

            with gr.Row():
                self.im_out_1 = gr.Image(type="pil", label="original")
                self.im_out_2 = gr.Image(type="pil", label="mask")
                self.im_out_3 = gr.Image(type="pil", label="composite")

            with gr.Row():
                # self.positive_prompt = gr.Textbox(label="Positive prompt")
                # self.negative_prompt = gr.Textbox(label="Negative prompt")
                self.iter_number = gr.Number(value=20, label="Steps")
                self.guidance_scale = gr.Number(value=0.7, label="Guidance Scale")
                self.enter_prompt = gr.Button("Enter prompt")

            with gr.Row():
                self.kandinsky_image = gr.Image(label="Kandinsky")
                self.stable_diffusion_image = gr.Image(label="Stable Diffusion")

            # Connect the UI and logic
            self.setup_settings.click(
                self.setup,
            )

            self.input_img.change(
                self.get_mask, 
                outputs=[self.im_out_1, self.im_out_2, self.im_out_3], 
                inputs=self.input_img,
            )

            # TODO: rewrite to cycle for each model
            self.enter_prompt.click(
                self.inpaint_image,
                # inputs=[self.positive_prompt, self.negative_prompt],
                inputs=[self.iter_number, self.guidance_scale],
                outputs=[self.kandinsky_image, self.stable_diffusion_image],
            )

     # Define the logic
    def setup(self):
        self.check_folders()
        self.start_logging()
        self.read_prompts()

    def get_mask(self, input_img):
        self.original_img = input_img["background"]
        mask = input_img["layers"][0]
        mask = np.array(Image.fromarray(np.uint8(mask)).convert("L"))
        self.masks = np.where(mask != 0, 255, 0)
        self.logger.info("New mask has been drawn")
        return [self.original_img, self.masks, input_img["composite"]]
    
    def prepare_input(self, image, mask):
        self.logger.info("Image shape: "+str(np.array(image).shape))
        self.logger.info("Mask shape: "+str(np.array(mask).shape))
        print(np.array(image).shape, np.array(mask).shape)

        image = Image.fromarray(np.uint8(image)).convert("RGB")
        w_orig, h_orig = image.size
        image = image.resize((512, 512))
        mask = Image.fromarray(np.uint8(mask)).resize((512, 512), Image.NEAREST)

        print(np.array(image).shape, np.array(mask).shape)
        self.logger.info("New image shape: "+str(np.array(image).shape))
        self.logger.info("New mask shape: "+str(np.array(mask).shape))
        return image, mask, w_orig, h_orig
    
    def generating_image(self, image, mask, iter_number, guidance_scale, w_orig, h_orig, model_name=""):
        for prompt in self.prompts:
            try:
                self.logger.info(f"Generate {model_name} with prompt: "+prompt)
                # TODO: write common AutoPipelineForInpainting for all models
                start_time = time.time()
                self.stable_diffusion_image = self.stable_diffusion.diffusion_inpaint(
                    image, mask, prompt, None, w_orig, h_orig, 
                    iter_number, guidance_scale,
                )
                curr_time = time.time()
                self.sd_avg_time += curr_time-start_time
                self.logger.info(f"{model_name} generated time: "+str(curr_time-start_time))
                self.save_img(self.stable_diffusion_image, f"{model_name}_"+prompt)
            except Exception as error:
                self.logger.info(f"ERROR WITH GENERATING IMAGE VIA {model_name}: "+error)
                print(f"ERROR WITH GENERATING IMAGE VIA {model_name}: ", error)

    def inpaint_image(self, iter_number, guidance_scale):
        image, mask, w_orig, h_orig = self.prepare_input(self.original_img, self.masks)

        self.generating_image(
            image, mask,
            iter_number, guidance_scale,
            w_orig, h_orig, 
            model_name="SD2",
        )

        self.logger.info(f"Turn TEXTUAL INVERSION ON")
        self.stable_diffusion.load_textual_inversion()

        self.generating_image(
            image, mask,
            iter_number, guidance_scale,
            w_orig, h_orig, 
            model_name="SD2_tuned",
        )

            # try:
            #     self.logger.info("Generate KAND with prompt: "+prompt)
            #     start_time = time.time()
            #     self.kandinsky_image = self.kandinsky.diffusion_inpaint(
            #         image, mask, prompt, None, w_orig, h_orig, 
            #         iter_number, guidance_scale,
            #     )
            #     curr_time = time.time()
            #     self.kandinsky_avg_time += curr_time-start_time
            #     self.logger.info("KANDINSKY generated time: "+str(curr_time-start_time))
            #     self.save_img(self.kandinsky_image, "KAND_"+prompt)
            # except Exception as error:
            #     self.logger.info("ERROR WITH GENERATING IMAGE VIA KANDINSKY: "+error)
            #     print("ERROR WITH GENERATING IMAGE VIA KANDINSKY: ", error)

        self.logger.info("DONE GENERATING IMAGES")
        self.logger.info("AVERAGE TIME FOR MODELS:")
        self.logger.info("Stable Diffusion: "+str(self.sd_avg_time/len(self.prompts)))
        # self.logger.info("Kandinsky: "+str(self.kandinsky_avg_time/len(self.prompts)))
        return self.kandinsky_image, self.stable_diffusion_image

    def save_img(self, img, prompt):
        im = Image.fromarray(img)
        path = os.path.join(self.path_to_output_imgs, prompt) + ".png"
        im.save(path)
        self.logger.info("Save image to "+path)
        print("SAVED: ", path)
    

window = GradioWindow()
