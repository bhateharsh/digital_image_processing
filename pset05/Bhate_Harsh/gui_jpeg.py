from tkinter import filedialog
from tkinter import *
import dippykit as dip
import numpy as np
from PIL import Image, ImageTk

class Window(Frame):

    filename="Bhate_Harsh/lena.png"
    c = 1

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # Creating entry button
        # int_var = FloatVar()
        e = Entry(self, width=5)
        e.pack()
        def callback():
            self.c = float(e.get())
        enterButton = Button(self, text="Enter c value", width=15, command=callback)
        
        # creating a button instance
        # quitButton = Button(self, text="Exit",command=self.showImg)
        chooseFile = Button(self, text="Upload File",command=self.loadImg)
        computeButton = Button(self, text="Find JPEG", command=self.computeDFT)
        # placing the button on my window
        e.place(x =0,y=0)
        enterButton.place(x=50,y = 0)
        chooseFile.place(x = (1150/2), y = 0)
        computeButton.place(x = (1150/2), y = 1150)
        # quitButton.place(x=0, y=0)

       

    def loadImg(self):
        self.filename = filedialog.askopenfilename()
        # load = Image.open(self.filename)
        # load = load.resize((500,500), Image.ANTIALIAS)
        # render = ImageTk.PhotoImage(load)
        # img = Label(self, image=render)
        # img.image = render
        # img.place(x=0, y=50)

    def computeDFT(self):
        #Convert Image to YCrCb
        im = dip.image_io.im_read(self.filename)
        im = dip.utilities.rgb2ycbcr(im) # HINT: Look into dip.rgb2ycbcr
        im = im[:,:,0]
        #Display Image
        img = Image.fromarray(im)
        img = img.resize((500,500), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(img)
        img = Label(self, image=render)
        img.image = render
        img.place(x=50, y=50)
        caption1 = Text(self, heigh=1,width=15)
        caption1.pack()
        caption1.insert(END, "Grayscale Image")
        caption1.place(x=50, y = 550)

        
        #Compute Entropy
        H = dip.metrics.entropy(im)
        str1 = "Entropy of the grayscale image: {:.2f} bits/pixel".format(H)
        
        #Huffman Encoding and Bit Rate Calculation
        byte_seq, _,_,_ = dip.coding.huffman_encode(im.flatten())
        l,b = im.shape
        im_bit_rate = (len(byte_seq)*8.0)/float(l*b)
        str2 = "Bit rate of the original image = {:.2f} bits/pixel".format(im_bit_rate)

        #DCT Calculation
        im = im.astype(float)
        im = im - 127.0
        block_size = (8, 8)
        im_DCT = dip.utilities.block_process(im, dip.dct_2d, block_size)
        
        #Display DCT
        img_DCT = Image.fromarray(im_DCT)
        img_DCT = img_DCT.resize((500,500), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(img_DCT)
        img = Label(self, image=render)
        img.image = render
        img.place(x=600, y=50)
        caption2 = Text(self, heigh=1,width=15)
        caption2.pack()
        caption2.insert(END, "DCT Image")
        caption2.place(x=600, y = 550)


        def step8(X):
            '''Apply X/(cQ), where X is a subblock'''
            Q_table = dip.JPEG_Q_table_luminance
            denominator = self.c*Q_table
            v = np.round(X/denominator).astype(int)
            return v
        im_DCT_quantized = dip.utilities.block_process(im_DCT, step8, block_size)

        # Display Quantized DCT Coefficients
        img_DCT_quantized = Image.fromarray(np.array(im_DCT_quantized),'RGB')
        img_DCT_quantized = img_DCT_quantized.resize((500,500), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(img_DCT_quantized)
        img = Label(self, image=render)
        img.image = render
        img.place(x=50, y=600)
        caption3 = Text(self, heigh=1,width=15)
        caption3.pack()
        caption3.insert(END, "Quantized Image")
        caption3.place(x=50, y = 1100)

        
        #Entropy Coding
        q_bit_stream, q_bit_stream_length, q_symbol_code_dict, _ = dip.huffman_encode(im_DCT_quantized.flatten())#.reshape(-1))
        q_bit_rate = q_bit_stream_length/float(l*b)
        str3 = "Bit rate of the compressed image = {:.2f} bits/pixel".format(q_bit_rate)

        #Saving the bitstream to a binary file
        bit_stream_file = open("CompressedSunset.bin", "wb")
        q_bit_stream.tofile(bit_stream_file)
        bit_stream_file.close()

        # Read the binary file
        bit_stream_file = open("CompressedSunset.bin", "rb")
        q_bit_stream = np.fromfile(bit_stream_file, dtype='uint8')
        bit_stream_file.close()

        # Decoding
        im_DCT_quantized_decoded = dip.huffman_decode(q_bit_stream,q_symbol_code_dict)
        im_DCT_quantized_decoded = im_DCT_quantized_decoded[:im.size]
        im_DCT_quantized_reconstructed = im_DCT_quantized_decoded.reshape(im.shape)

        # Dequantization
        def step12(X):
            '''Apply X*(cQ), where X is a subblock'''
            Q_table = dip.JPEG_Q_table_luminance
            numerator = self.c*Q_table 
            return np.round(np.multiply(X,numerator)).astype(int)

        im_DCT_reconstructed = dip.utilities.block_process(im_DCT_quantized_reconstructed, step12, block_size)  

        # Inverse DCT
        im_reconstructed = dip.utilities.block_process(im_DCT_reconstructed, dip.transforms.idct_2d, block_size)

        # Add 127 to every pixel
        im_reconstructed = im_reconstructed + 127

        # Display Reoconstructed Image
        img_reconstructed = Image.fromarray(np.array(im_reconstructed))
        img_reconstructed = img_reconstructed.resize((500,500), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(img_reconstructed)
        img = Label(self, image=render)
        img.image = render
        img.place(x=600, y=600)
        caption4 = Text(self, heigh=1,width=25)
        caption4.pack()
        caption4.insert(END, "Reconstructed Image")
        caption4.place(x=600, y = 1100)

        im = im + 127

        # Calculating MSE and PSNR
        MSE = dip.metrics.MSE(im, im_reconstructed)
        PSNR = dip.metrics.PSNR(im_reconstructed, im, 256)

        str4 = "MSE = {:.2f}".format(MSE)
        str5 = "PSNR = {:.2f} dB".format(PSNR)

        #Display
        final_str = str1+"\n"+str2+"\n"+str3+"\n"+str4+"\n"+str5
        entropyGrayscale = Text(root, heigh=5,width=50)
        entropyGrayscale.pack()
        entropyGrayscale.insert(END, final_str)
        entropyGrayscale.place(x=50, y = 1200)



      
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

root.geometry("1150x1375")

#creation of an instance
app = Window(root)

#mainloop 
root.mainloop()