import gradio as gr

def predict(image, description, milage):
    return f"has_image: {image is not None} | description: {description} | milage: {milage}"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Car Photo"),
        gr.Textbox(label="Description", lines=3),
        gr.Textbox(label="Mileage", placeholder="e.g. 45000")
    ],
    outputs="text",
    title="Test"
).launch()
