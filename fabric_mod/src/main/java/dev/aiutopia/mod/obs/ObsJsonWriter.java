package dev.aiutopia.mod.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

/** Tiny helper to keep ObservationBuilder code readable. Wraps common
 *  patterns of building JsonObject / JsonArray from numeric primitives. */
public final class ObsJsonWriter {
    private ObsJsonWriter() {}

    /** 1-element JSON array — used for Box((1,)) gym fields that gym still
     *  expects as a shape-(1,) ndarray on the Python side. Wrapping at the
     *  Java boundary avoids a special-case in Python's _decode_obs. */
    public static JsonArray vec1(double x) {
        JsonArray a = new JsonArray();
        a.add(x);
        return a;
    }
    public static JsonArray vec(double x, double y, double z) {
        JsonArray a = new JsonArray();
        a.add(x); a.add(y); a.add(z);
        return a;
    }
    public static JsonArray vec2(double x, double y) {
        JsonArray a = new JsonArray();
        a.add(x); a.add(y);
        return a;
    }
    public static JsonArray intArray(int[] xs) {
        JsonArray a = new JsonArray();
        for (int x : xs) a.add(x);
        return a;
    }
    public static JsonArray floatArray(float[] xs) {
        JsonArray a = new JsonArray();
        for (float x : xs) a.add(x);
        return a;
    }
    public static JsonObject withScalar(String key, double value) {
        JsonObject o = new JsonObject();
        o.add(key, new JsonPrimitive(value));
        return o;
    }
}
