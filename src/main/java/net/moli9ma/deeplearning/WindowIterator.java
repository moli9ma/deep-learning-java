package net.moli9ma.deeplearning;


import java.util.Iterator;

public class WindowIterator implements Iterable<Window>, Iterator<Window> {

    // 可動域
    private final int maxX;
    private final int maxY;

    // Window幅&高さ
    private final int height;
    private final int width;

    // stride
    private final int strideX;
    private final int strideY;

    // Window
    private Window window;

    public WindowIterator(int maxX, int maxY, int height, int width, int strideX, int strideY) {
        this.maxX = maxX;
        this.maxY = maxY;
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
    }

    public Iterator<Window> iterator() {
        return this;
    }

    public boolean hasNext() {
        if (this.window == null) {
            return true;
        }

        if (isWindowEndOfX() && isWindowEndOfY()) {
            return false;
        }

        return true;
    }

    public Window next() {
        if (this.window == null) {
            this.window = new Window(width, height);
            return this.window;
        }

        if (!isWindowEndOfX()) {
            Window window = new Window(
                    this.window.startX + strideX,
                    this.window.endX + strideX,
                    this.window.startY,
                    this.window.endY);

            this.window = window;
            return window;
        } else {
            Window window = new Window(
                    0,
                    this.width,
                    this.window.startY + strideY,
                    this.window.endY + strideY);

            this.window = window;
            return window;
        }
    }

    public void remove() {
        // do nothing
    }

    private boolean isWindowEndOfX() {
        return this.window.endX == this.maxX;
    }

    private boolean isWindowEndOfY() {
        return this.window.endY == this.maxY;
    }
}
