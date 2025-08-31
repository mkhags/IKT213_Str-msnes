import cv2
import os

def skriv_bilde_info(bilde):
    m = bilde.shape; h = m[0]; b = m[1]; k = m[2];
    print(f"Høyde: {h}\nBredde: {b}\nKanaler: {k}\nStørrelse: {bilde.size}\nDatatype: {bilde.dtype}");

def lagre_kamera_info():
    cam = cv2.VideoCapture(0);
    fps = cam.get(cv2.CAP_PROP_FPS);
    if fps == 0: fps = 30;

    b = cam.get(cv2.CAP_PROP_FRAME_WIDTH);
    h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT);

    mappe = "solutions";
    os.makedirs(mappe, exist_ok=True);

    fil = os.path.join(mappe, "camera_outputs.txt");
    with open(fil, "w") as f:
        f.write(f"fps: {int(fps)}\n");
        f.write(f"høyde: {int(h)}\n");
        f.write(f"bredde: {int(b)}\n");

    print(f"Kamera-info lagret i {fil}");
    cam.release();

def main():
    bilde = cv2.imread("lena-1.png");
    skriv_bilde_info(bilde);
    lagre_kamera_info();

if __name__ == "__main__":
    main();