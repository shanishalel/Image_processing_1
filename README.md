# Image_processing_ex1
In this exercise:
* I used Phyton version 3.7
* I used PyCharm by Jet Brains

### The files that I submitting are : 
* ex1_utils - the file that include all the function that we ask for 
* gamma - the file that incluse the gamma correction function 
* ex1_main - the file that include all the calling to the function and the tests also

#### ex1_utils:
* imReadAndConvert - Reads an image, and returns the image converted as requested. 
this function return the image as np array . 
* imDisplay - Reads an image as RGB or GRAY_SCALE and displays it by using plt.show()
* transformRGB2YIQ - Converts an RGB image to YIQ color space by multiply the img and matrix
[[0.299, 0.587, 0.114],
[0.596, -0.275, -0.321],
[0.212, -0.523, 0.311]]
* transformYIQ2RGB - Converts an YIQ image to RGB color space by multiply the yiq img with the inverse matrix
of the same matrix as one before
* hsitogramEqualize - Equalizes the histogram of an image by using the
* quantizeImage - Quantized an image in to **nQuant** colors and return (List[qImage_i],List[error_i])
* init_bound - function that init the boundary as we study from tirgul
* finding_error -  function that found the value of the error - from the tirgul , mse= sqrt(sum(imgold-imgnew)^2)/allpixels
* finding_q -  function that finding the q by :(z[i] to z[i+1] : g*h(g))/(z[i] to z[i+1] h(g))
* fix_z - function that found the z according to : z[i]=q[i-1]-q[i]\2
  
#### ex1_main:
* histEqDemo 
* quantDemo
  
#### gamma :
* on_trackbar - function that do nothing
* gammaDisplay - GUI for gamma correction (by tracking bar that is from 0 to 10) 

### Tests:
I add 2 photos test1 and test2 that are my tests photos. 
I write the code for my tests in the main and put it under comment so it will 
not bother. test1 is picture of mona liza I choose it because I want to see if we can change the picture so we will get picture that is more sharpen that the 
original paint . test2 is picture of lion that is created from alot of paint and I want to see how the quantizition of this picture will look like. 
