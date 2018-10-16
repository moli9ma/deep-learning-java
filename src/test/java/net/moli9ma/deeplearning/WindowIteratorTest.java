package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class WindowIteratorTest {

    @Test
    void windowIterator() {
        {
            int width = 3;
            int height = 3;
            int kernelWidth = 2;
            int kernelHeight = 2;
            int strideX = 1;
            int strideY = 1;

            WindowIterator iterator = new WindowIterator (
                    width,
                    height,
                    kernelWidth,
                    kernelHeight,
                    strideX,
                    strideY
            );

            LinkedList<Window> queue = new LinkedList<Window>();
            queue.offer(new Window(0,2, 0, 2));
            queue.offer(new Window(1,3, 0, 2));
            queue.offer(new Window(0,2, 1, 3));
            queue.offer(new Window(1,3, 1, 3));

            while (iterator.hasNext()) {
                Window window = iterator.next();
                Window expected = queue.poll();
                assertEquals(expected.getStartX(), window.getStartX());
                assertEquals(expected.getStartY(), window.getStartY());
                assertEquals(expected.getEndX(), window.getEndX());
                assertEquals(expected.getEndY(), window.getEndY());
            }
        }

        {
            int width = 4;
            int height = 3;
            int kernelWidth = 2;
            int kernelHeight = 2;
            int strideX = 1;
            int strideY = 1;

            WindowIterator iterator = new WindowIterator (
                    width,
                    height,
                    kernelWidth,
                    kernelHeight,
                    strideX,
                    strideY
            );

            LinkedList<Window> queue = new LinkedList<Window>();
            queue.offer(new Window(0,2, 0, 2));
            queue.offer(new Window(1,3, 0, 2));
            queue.offer(new Window(2,4, 0, 2));
            queue.offer(new Window(0,2, 1, 3));
            queue.offer(new Window(1,3, 1, 3));
            queue.offer(new Window(2,4, 1, 3));

            while (iterator.hasNext()) {
                Window window = iterator.next();
                Window expected = queue.poll();
                assertEquals(expected.getStartX(), window.getStartX());
                assertEquals(expected.getStartY(), window.getStartY());
                assertEquals(expected.getEndX(), window.getEndX());
                assertEquals(expected.getEndY(), window.getEndY());
            }
        }

        {
            int width = 6;
            int height = 4;
            int kernelWidth = 2;
            int kernelHeight = 2;
            int strideX = 2;
            int strideY = 2;

            WindowIterator iterator = new WindowIterator (
                    width,
                    height,
                    kernelWidth,
                    kernelHeight,
                    strideX,
                    strideY
            );

            LinkedList<Window> queue = new LinkedList<Window>();
            queue.offer(new Window(0,2, 0, 2));
            queue.offer(new Window(2,4, 0, 2));
            queue.offer(new Window(4,6, 0, 2));
            queue.offer(new Window(0,2, 2, 4));
            queue.offer(new Window(2,4, 2, 4));
            queue.offer(new Window(4,6, 2, 4));

            while (iterator.hasNext()) {
                Window window = iterator.next();
                Window expected = queue.poll();
                assertEquals(expected.getStartX(), window.getStartX());
                assertEquals(expected.getStartY(), window.getStartY());
                assertEquals(expected.getEndX(), window.getEndX());
                assertEquals(expected.getEndY(), window.getEndY());
            }
        }
    }
}