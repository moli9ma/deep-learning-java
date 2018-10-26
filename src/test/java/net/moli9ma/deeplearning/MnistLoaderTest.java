package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import static net.moli9ma.deeplearning.MnistLoader.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MnistLoaderTest {

    @Test
    void test() throws IOException {
        MnistLoader train = new MnistLoader(TrainImages, TrainLabels);
        MnistLoader test = new MnistLoader(TestImages, TestLabels);
        assertEquals(60000, train.size);
        assertEquals(784, train.imageSize);
        assertEquals(10000, test.size);
        assertEquals(784, test.imageSize);

        // 訓練データの先頭の100イメージをPNGとして出力します。
/*
        if (!TrainImagesOutput.exists()) TrainImagesOutput.mkdirs();
        for (int i = 0; i < 100; ++i) {
            File image = new File(TrainImagesOutput, String.format("%05d-%d.png", i, train.label(i)));
            train.writePngFile(i, image);
        }
*/
    }

    @Test
    void test2() {
        byte[][] input = new byte[3][3];
        input[0] = new byte[]{1,2,3};
        input[1] = new byte[]{1,2,3};
        input[2] = new byte[]{1,2,3};

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double result1e = input[i][j] / 255.0;
                System.out.println(result1e);

                System.out.println("xx");

                double result = (input[i][j] & 0xff) / 255.0;
                System.out.println(result);
            }
        }
    }



}
