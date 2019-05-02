package com.hci.project.malayalamtutions.models;

public interface Classifier {
    String name();

    Classification recognize(final float[] pixels);
}
