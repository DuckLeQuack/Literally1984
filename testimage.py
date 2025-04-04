from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
import torch
from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  



#Check this https://huggingface.co/microsoft/Florence-2-large-ft
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer


def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.show()  




# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(r"images/video frame 10 and 30.png").convert('RGB')

# task_prompt = '<CAPTION>'
# a = run_example(task_prompt)

# task_prompt = '<DETAILED_CAPTION>'
# a = run_example(task_prompt)

# # task_prompt = '<MORE_DETAILED_CAPTION>'
# # a = run_example(task_prompt)
# print(a)

#task_prompt = '<OD>' #Object detection
task_prompt = '<DENSE_REGION_CAPTION>'
results = run_example(task_prompt)
print(results)
plot_bbox(image, results['<DENSE_REGION_CAPTION>'])

#plot_bbox(image, results['<OD>'])



#OCR
# url = "http://ecx.images-amazon.com/images/I/51UUzBDAMsL.jpg?download=true"
#image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
# image = Image.open(r"picrel.png").convert('RGB')

# task_prompt = '<OCR_WITH_REGION>'
# results = run_example(task_prompt)
# print(results)





def draw_ocr_bboxes(image, prediction, scale=1):
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",
        
                    fill=color)
    image.show()

# output_image = copy.deepcopy(image)
# w, h = output_image.size
# scale = 800 / max(w, h)
# new_output_image = output_image.resize((int(w * scale), int(h * scale)))
# draw_ocr_bboxes(new_output_image, results['<OCR_WITH_REGION>'], scale=scale)  


