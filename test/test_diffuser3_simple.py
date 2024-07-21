from panna import Diffuser3, save_image

model = Diffuser3()
output = model.text2image(["A majestic lion jumping from a big stone at night"], batch_size=1)
save_image(output[0], "./test/test_images/sample.png")
