import cv2
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime



width = 640 #larghezza standard raspberry
height = 480 # altezza standard raspberry
def bozza_test_camera():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        dim = (width,height)
        frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
        #frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_ISOLATED, value=[0, 200, 200])
        #cv2.rectangle(frame,(640,140),(0,340),(0,255,0),4)   #600 indica l'altezza destra del rettangolo 200 la base al basso del rettangolo e gli altri due i restanti
        #cv2.rectangle(frame, (440, 140), (200, 340), (0, 255, 0), 4)
        cv2.circle(frame, (320, 240), 70, (0, 255, 0), 2)
        if not ret:
            print("Impossibile catturare l'immagine")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC per uscire
            print("Chiusura schermata")
            break
        elif k == ord(' '):
            # SPACE per scattare la foto
            now = datetime.now()
            date = now.strftime("%Y-%m-%d_at")

            time = now.strftime("_%H.%M.%S")#rinomino la foto in base alla data di oggi e all'orario
            mese_orario= date + time
            img_name = 'analysis_of_{}.png'.format(mese_orario)
            copy_name =img_name
            cv2.imwrite(img_name, frame)
            CutPhoto(copy_name)
            print("{} written!".format(img_name))
            cattura = cv2.imread("result.png")
            img_processing_image(cattura,1)#metto 1 solo perchè è un testing, poi non servirà più

            # break  mettendo questo ti scatta solo una foto e poi ti chiude la finestra direttamente, l'ho tolto cosi è possibile fare più scatti
    cam.release()
    cv2.destroyAllWindows()
def img_processing_image(img2, i):
    risultato = str(i)
    if(i == 10):
        risultato = " testing"
    lab_image = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab_image) #Divisione di ogni riga di pixel nei 3 valore del LAB
    #print("LUCE")
    #print(l_channel)
    rows = len(a_channel) #righe pixel valutate in cromatura a
    columns = len(a_channel[0]) #colonne pixel valutate in cromatura a
    n_pixel = rows * columns  # numero totale pixel
    i=0

    a_filtrate=np.empty((n_pixel))
    for x in range(0,rows):
       for y in range(0,columns):
           if a_channel[x][y] != 42 :#42 è il colore verde sulla cromatura a* di opencv( in realta dovrebbe essere -128)
               if a_channel[x][y] < 128 or a_channel[x][y] > 148:
                a_filtrate[i] = a_channel[x][y]
                i= i + 1


    a_palpebra=a_filtrate[0:i]#matrice della palpebra
    a_star= np.average(a_palpebra)
    a_star=(a_star-128) #moltiplico per 2 perchè cosi è vicino al valore dell hb

    print("A STAR NEL VETTORE FILTRATO")
    print(a_star)
    a = np.average(a_channel)  # valore medio di a, che dovrebbe (credo) essere il valore che ci serve
    a = a - 128  # valore preciso di a*
    print("A STAR SENZA FILTRAGGIO")
    print(a)
    print("N ELEMENTI FILTRATI")
    print(i)
    print("NUMERO PIXEL TOTALI")
    print(n_pixel)
   # print("A CROMATURA")
   # print (a_channel)
   # print("B CROMATURA")
   # print( b_channel)
   # b=np.average(b_channel)
   # print(b)

    cv2.imshow("Result"+risultato, img2)
def CutPhoto(src):#ritaglia il suddetto riquadro verde dell'immagine

    #display(img)
    im = Image.open(src)
    #im1 = im.crop((204, 144, 435, 335))  # left,top,right,bottom
    im1 = im.crop((260, 178, 381, 299))
    im1.save(src)
    img = Image.open(src)
    #display("a",img)

    height, width = img.size
    lum_img = Image.new('L', [height, width], 0)

    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0, 0), (height, width)], 0, 180,
                  fill=255, outline="white")
    img_arr = np.array(img)
    lum_img_arr = np.array(lum_img)
    #display(Image.fromarray(lum_img_arr))
    final_img_arr = np.dstack((img_arr, lum_img_arr))
    a = Image.fromarray(final_img_arr)
    a.save(src)
    rgba = np.array(Image.open(src))

    # Make image transparent white anywhere it is transparent
    rgba[rgba[..., -1] == 0] = [255, 255, 255,
                                255]  # inserisco lo sfondo verde ma anche con sfondo bianco funziona discretamente

    # Make back into PIL Image and save
    Image.fromarray(rgba).save('result.png')
   # b = cv2.cvtColor("PROVA.PNG", cv2.COLOR_RGBA2RGB)


    #im1 = im.crop((204, 144, 435, 335))  # left,top,right,bottom

def SlicAlgoritm(img):
    # gaussian blur
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=20.0)
    slic.iterate(10)  # Number of iterations, the greater the better
    mask_slic = slic.getLabelContourMask()  # Get Mask, Super pixel edge Mask==1
    label_slic = slic.getLabels()  # Get superpixel tags
    number_slic = slic.getNumberOfSuperpixels()  # Get the number of super pixels
    mask_inv_slic = cv2.bitwise_not(mask_slic)
    img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic)  # Draw the superpixel boundary on the original image
    cv2.imshow("img_slic", img_slic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img2 = cv2.imread('proviamo2.JPG')
#img_processing_image(img2,10)
#SlicAlgoritm(img2)
bozza_test_camera()