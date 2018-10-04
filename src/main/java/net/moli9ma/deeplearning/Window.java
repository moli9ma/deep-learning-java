package net.moli9ma.deeplearning;

public class Window {

    // 現在のX軸範囲
    int startX;
    int endX;

    // 現在のY軸範囲
    int startY;
    int endY;

    public Window(int startX, int endX, int startY, int endY) {
        this.startX = startX;
        this.endX = endX;
        this.startY = startY;
        this.endY = endY;
    }

    public Window(int width, int height) {
        this.startX = 0;
        this.endX = width;
        this.startY = 0;
        this.endY = height;
    }

    public int getStartX() {
        return startX;
    }

    public int getEndX() {
        return endX;
    }

    public int getStartY() {
        return startY;
    }

    public int getEndY() {
        return endY;
    }



}
