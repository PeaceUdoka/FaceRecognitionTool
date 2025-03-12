import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import cv2
import numpy as np
import io

kivy.require('2.0.0')  # Or your Kivy version

class CameraKivyApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.img = None  # Store the current frame
        self.ksize = 10  # Initial kernel size for blur

    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Image Widget
        self.img_widget = Image()
        self.layout.add_widget(self.img_widget)

        # Slider for Kernel Size
        self.slider = Slider(min=1, max=100, value=self.ksize)
        self.slider.bind(value=self.on_slider_value)
        self.layout.add_widget(self.slider)

        # Schedule Camera Update
        Clock.schedule_interval(self.update, 1.0/30.0) # 30 fps

        return self.layout

    def on_slider_value(self, instance, value):
        self.ksize = int(value) # Update kernel size

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.img = frame # Store the frame

            # Apply Blur
            blurred_img = cv2.blur(self.img, (self.ksize, self.ksize))

            # Convert to Kivy Texture
            buf = cv2.flip(blurred_img, 0).tostring()  # Flip for correct orientation
            texture = Texture.create(size=(self.img.shape[1], self.img.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Update Image Widget
            self.img_widget.texture = texture

    def on_stop(self):
        # Release the camera when the app closes
        self.capture.release()

if __name__ == '__main__':
    CameraKivyApp().run()
