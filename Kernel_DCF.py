import cv2
import src.canny as Can
import src.SURF as SURF
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def canny_multi():
    img_path = ("H:/data/crop/")
    piclist = os.listdir(img_path)
    for i in tqdm(range(len(piclist))):
        img = cv2.imread(img_path+piclist[i])
        # CANNY_edge_extract
        CANNY_img, resize_img = Can.img_canny(img)
        # Surf_keypoint_extract
        Surf_img, kp = SURF.SURF_Pro(CANNY_img)
        line_img = SURF.draw_horizonline(kp, resize_img)
        # # show img
        # show_img = line_img
        cv2.imwrite(img_path+"picture/"+piclist[i],line_img)
        pass

def canny(num):
    img_path = ("H:/data/crop/")

    img_num = str(num)+".jpg"

    img = cv2.imread(img_path+img_num)

    # CANNY_edge_extract
    CANNY_img,resize_img = Can.img_canny(img)

    # Surf_keypoint_extract
    Surf_img,kp = SURF.SURF_Pro(CANNY_img)
    line_img = SURF.draw_horizonline(kp,resize_img)

    # # show img
    # show_img = line_img

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(CANNY_img)
    plt.subplot(2,2,2)
    plt.imshow(Surf_img)
    plt.subplot(2,2,3)
    plt.imshow(line_img)

    plt.show()

    # cv2.imshow("canny",show_img)
    # k = cv2.waitKey(0) & 0xFF
    # if k == ord("q"):
    #     cv2.destroyAllWindows()

def printMenu(title, opts):
    print ("\n+---------------------------------------------- +")
    print("|  {0:42}   |\n".format(title))
    print("+---------------------------------------------- +")
    for i in range(0,len(opts)):
        print("| {0:2}. {1:40} |".format(i + 1, opts[i]))
    print ("+---------------------------------------------- +")
    return int(input("Please select and option: "))



if __name__ == '__main__':
    print("please input your path to the data")
    mainOpts = [ "single picture", "multi picture" ]

    while True:
        opt = printMenu("what would you like to do?", mainOpts)
        if opt == 1:
            num = input("picture number:")
            canny(num)
        elif opt == 2:
            canny_multi()
        else:
            print("Unrecognized option.")

    # num = input("picture number:")
    # canny(num)