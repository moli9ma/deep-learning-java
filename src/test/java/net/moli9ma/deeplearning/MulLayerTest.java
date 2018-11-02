package net.moli9ma.deeplearning;

import net.moli9ma.deeplearning.layer.MulLayer;
import net.moli9ma.deeplearning.layer.MulLayerData;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MulLayerTest {


    @Test
    void mulLayerTest() {


        {
            double apple = 100;
            double apple_num = 2;
            double tax = 1.1;


            MulLayer mulAppleLayer = new MulLayer();
            MulLayer mulTaxLayer = new MulLayer();

            MulLayerData appleLayerData = new MulLayerData(apple, apple_num);
            double applePrice = mulAppleLayer.forward(appleLayerData);

            MulLayerData taxLayerData = new MulLayerData(applePrice, tax);
            double price = mulTaxLayer.forward(taxLayerData);

            assertEquals(220, (int) price);


            int dprice = 1;

            MulLayerData data = mulTaxLayer.backward(dprice);
            double dApplePrice = data.x;
            double dTax = data.y;
            assertEquals(1.1, dApplePrice);
            assertEquals(200, dTax);


            MulLayerData dAppleLayerData = mulAppleLayer.backward(dApplePrice);
            double dApple = dAppleLayerData.x;
            double dAppleNum = dAppleLayerData.y;
            assertEquals(2.2, dApple);
            assertEquals(110, (int)dAppleNum);


        }


    }



}
