import os
import numpy as np
import torch
from PIL import Image
import gradio as gr
# from .stable_diffusion import StableDiffusionModel
from .sd_inpaint_dreambooth import StableDiffusionModel
# from .kandinsky import KandinskyModel
# from .kandinsky_3 import Kandinsky3Model
import logging
import time 

class GradioWindow():
    def __init__(self) -> None:
        self.path_to_orig_imgs = None
        self.path_to_output_imgs = None
        self.path_to_prompts = None
        self.path_to_negative_prompts = None
        self.path_to_logs = None
        self.path_to_ti = None
        self.path_to_db = None

        self.original_img = None
        self.masks = None
        self.prompts = None
        self.logger = None
        self.stable_diffusion = None 
        self.kandinsky = None 

        #TODO: rewrite time calculating to something goodx
        self.sd_avg_time = 0
        self.kandinsky_avg_time = 0

        self.folders = []

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
        self.stable_diffusion = StableDiffusionModel(textual_inversion_checkpoint=self.path_to_ti, 
                                                    dreambooth_checkpoint=self.path_to_db)
        # self.kandinsky = KandinskyModel()
        # self.kandinsky = Kandinsky3Model()
        self.logger.info("Models loaded")
        print("Models loaded!")

    def main(self):
        with gr.Blocks() as self.demo:
            with gr.Row():
                self.input_img = gr.ImageEditor(
                    type="pil",
                    label="Input",
                )
                with gr.Column():
                    self.orig_imgs = gr.Textbox(label="Images path", value="images/left_label/left")
                    self.output_imgs = gr.Textbox(label="Output path", value="images/output_imgs/K3")
                    self.prompts = gr.Textbox(label="Prompt path", value="prompts/pothole.txt")
                    self.negative_prompts = gr.Textbox(label="Negative prompt path", value="images/negative_prompts.txt")
                    self.logs = gr.Textbox(label="Logs file name", value="out.log")
                    self.ti = gr.Textbox(label="Textual inversion path", value="model_output/SD2inp_ti_cat-avocado/checkpoint-6000")
                    self.db = gr.Textbox(label="DreamBooth path", value="model_output/db_pothole")
                    self.setup_settings = gr.Button("Set up")

            with gr.Row():
                self.im_out_1 = gr.Image(type="pil", label="original")
                self.im_out_2 = gr.Image(type="pil", label="mask")
                self.im_out_3 = gr.Image(type="pil", label="composite")

            with gr.Row():
                 with gr.Column():
                    self.iter_number = gr.Number(value=20, label="Steps")
                    self.guidance_scale = gr.Number(value=0.7, label="Guidance Scale")
                 with gr.Column():
                    self.button_load_models = gr.Button("Load models") 
                    self.button_enter_prompt = gr.Button("Enter prompt")     

            with gr.Row():
                self.kandinsky_image = gr.Image(label="Kandinsky")
                self.stable_diffusion_image = gr.Image(label="Stable Diffusion")

            # Connect the UI and logic
            self.setup_settings.click(
                self.setup,
                inputs=[
                    self.orig_imgs,
                    self.output_imgs,
                    self.prompts,
                    self.negative_prompts,
                    self.logs,
                    self.ti,
                    self.db,
                ],
            )

            self.input_img.change(
                self.get_mask, 
                outputs=[self.im_out_1, self.im_out_2, self.im_out_3], 
                inputs=self.input_img,
            )

            self.button_load_models.click(
                self.load_models,
            )

            # TODO: rewrite to cycle for each model
            self.button_enter_prompt.click(
                self.inpaint_image,
                inputs=[self.iter_number, self.guidance_scale],
                outputs=[self.kandinsky_image, self.stable_diffusion_image],
            )

     # Define the logic
    def setup(self, path_to_orig_imgs, path_to_output_imgs, 
              path_to_prompts, path_to_negative_prompts, 
              path_to_logs, path_to_ti, path_to_db):
        
        print("START SETUP")
        self.path_to_orig_imgs = path_to_orig_imgs
        self.path_to_output_imgs = path_to_output_imgs
        self.path_to_prompts = path_to_prompts
        self.path_to_negative_prompts = path_to_negative_prompts
        self.path_to_logs = path_to_logs
        self.path_to_ti = path_to_ti
        self.path_to_db = path_to_db
        

        self.folders = [self.path_to_orig_imgs, self.path_to_output_imgs]

        self.check_folders()
        self.start_logging()
        self.read_prompts()

        print("START DONE")

    def get_mask(self, input_img):
        self.original_img = input_img["background"]
        mask = input_img["layers"][0]
        mask = np.array(Image.fromarray(np.uint8(mask)).convert("L"))
        self.masks = np.where(mask != 0, 255, 0)
        self.logger.info("New mask has been drawn")
        return [self.original_img, self.masks, input_img["composite"]]
    
    def get_bbox(self, mask):
        mask = np.array(mask)
        non_zero_coords = np.argwhere(mask == 255)

        y1, x1 = non_zero_coords.min(axis=0)
        y2, x2 = non_zero_coords.max(axis=0)

        padding = 20
        x1_padded = max(x1 - padding, 0)
        y1_padded = max(y1 - padding, 0)
        x2_padded = min(x2 + padding, mask.shape[1] - 1)
        y2_padded = min(y2 + padding, mask.shape[0] - 1)

        bounding_box_with_padding = [x1_padded, y1_padded, x2_padded, y2_padded]
        print("bounding_box_with_padding", bounding_box_with_padding)
        return bounding_box_with_padding

    def prepare_input(self, image, mask):
        self.logger.info("Image shape: "+str(np.array(image).shape))
        self.logger.info("Mask shape: "+str(np.array(mask).shape))
        print(np.array(image).shape, np.array(mask).shape)

        image = Image.fromarray(np.uint8(image)).convert("RGB")
        w_orig, h_orig = image.size
        image = image.resize((512, 512))
        mask = Image.fromarray(np.uint8(mask)).resize((512, 512), Image.NEAREST)
        print("MASK type: ", type(mask))

        print(np.array(image).shape, np.array(mask).shape)
        bbox = self.get_bbox(mask)
        left, upper, right, lower = bbox
        # Crop the image and mask
        cropped_image = image.crop((left, upper, right, lower))
        cropped_mask = mask.crop((left, upper, right, lower))

        self.logger.info("New image shape: "+str(np.array(image).shape))
        self.logger.info("New mask shape: "+str(np.array(mask).shape))
        return image, mask, w_orig, h_orig, cropped_image, cropped_mask, bbox
    
    def generating_image(self, diffusion_model, image, mask, 
                            cropped_image, cropped_mask,
                            iter_number, guidance_scale, 
                            w_orig, h_orig, bbox, model_name=""):
        
        for prompt in self.prompts:
            try:
                self.logger.info(f"Generate {model_name} with prompt: "+prompt)
                start_time = time.time()

                generated_image = diffusion_model.diffusion_inpaint(
                    # image, mask, prompt, None, w_orig, h_orig, 
                    cropped_image, cropped_mask, prompt, None, w_orig, h_orig, 
                    iter_number, guidance_scale,
                )
                curr_time = time.time()
                # inpaint_image = generated_image

                # Calculate the size of the bounding box
                left, upper, right, lower = bbox
                bounding_box_width = right - left + 1
                bounding_box_height = lower - upper + 1
                replace_img_resized = generated_image.resize((bounding_box_width, bounding_box_height))
                # replace_img_resized.save("2.jpg")

                image.paste(replace_img_resized, (bbox[0], bbox[1]))
                inpaint_image = image
                inpaint_image = inpaint_image.resize((w_orig, h_orig))
                inpaint_image = np.array(inpaint_image)

                self.sd_avg_time += curr_time-start_time
                self.logger.info(f"{model_name} generated time: "+str(curr_time-start_time))
                self.save_img(inpaint_image, f"{model_name}_"+prompt)
            except Exception as error:
                self.logger.info(f"ERROR WITH GENERATING IMAGE VIA {model_name}: "+str(error))
                print(f"ERROR WITH GENERATING IMAGE VIA {model_name}: ", error)

    def inpaint_image(self, iter_number, guidance_scale):
        image, mask, w_orig, h_orig, cropped_image, cropped_mask, bbox = self.prepare_input(self.original_img, self.masks)

        self.generating_image(
            self.stable_diffusion,
            # self.kandinsky,
            image, mask,
            cropped_image, cropped_mask,
            iter_number, guidance_scale,
            w_orig, h_orig, bbox,
            model_name="SD2",
        )

        # # self.logger.info(f"TURN DREAMBOOTH ON")
        # self.logger.info(f"Turn TEXTUAL INVERSION ON")
        # self.stable_diffusion.load_textual_inversion(self.path_to_ti)

        # self.sd_generating_image(
        #     self.stable_diffusion,
        #     image, mask,
        #     iter_number, guidance_scale,
        #     w_orig, h_orig,  
        #     model_name="SD2_ti-inp",
        # )

        # self.logger.info(f"Turn TEXTUAL INVERSION OFF")
        # self.stable_diffusion.unload_textual_inversion()

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
