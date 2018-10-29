package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AddLayerTest {

    @Test
    void addLayerTest() {
        AddLayer addLayer = new AddLayer();
        AddLayerData data = new AddLayerData(10, 10);
        double result = addLayer.foraward(data);
        assertEquals(20, result);

        AddLayerData dAddLayerData = addLayer.backward(1.1);
        assertEquals(1.1, dAddLayerData.x);
        assertEquals(1.1, dAddLayerData.y);
    }

}
