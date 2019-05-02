package com.hci.project.malayalamtutions;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.PointF;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.hci.project.malayalamtutions.models.Classification;
import com.hci.project.malayalamtutions.models.Classifier;
import com.hci.project.malayalamtutions.models.TensorFlowClassifier;
import com.hci.project.malayalamtutions.views.DrawModel;
import com.hci.project.malayalamtutions.views.DrawView;
import com.hci.project.malayalamtutions.utils.utililities;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements View.OnClickListener, View.OnTouchListener  {

    private static final int PIXEL_WIDTH = 32;

    // ui elements
    private Button clearBtn, classBtn;
    private TextView resText;
    private List<Classifier> mClassifiers = new ArrayList<>();

    // views
    private DrawModel drawModel;
    private DrawView drawView;
    private PointF mTmpPiont = new PointF();

    private float mLastX;
    private float mLastY;

    public int selected;
    private AssetManager assetManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        assetManager = getAssets();

        drawView = (DrawView) findViewById(R.id.draw);

        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

        drawView.setModel(drawModel);
        drawView.setOnTouchListener(this);

        clearBtn = (Button) findViewById(R.id.clear);
        clearBtn.setOnClickListener(this);

        classBtn = (Button) findViewById(R.id.check);
        classBtn.setOnClickListener(this);

        resText = (TextView) findViewById(R.id.typer);

        loadModel();
        selected = utililities.getRandomClasses();
        String s = utililities.getCharsFromIndex(utililities.getIndexOfClass(selected));
        resText.setText(s);
    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.clear) {
            drawModel.clear();
            drawView.reset();
            drawView.invalidate();
        } else if (v.getId() == R.id.check) {
            float pixels[] = drawView.getPixelData();

            String text = "";
            for (Classifier classifier : mClassifiers) {
                final Classification res = classifier.recognize(pixels);
                if (res.getLabel() == null) {
                    text += classifier.name() + ": ?\n";
                } else {
                    text += String.format("%s: %s, %f\n", classifier.name(), res.getLabel(),
                            res.getConf());
                }
            }
            //Toast.makeText(getApplicationContext(),text,Toast.LENGTH_LONG).show();
            if(text.contains(""+selected)){
                Toast.makeText(getApplicationContext(),"CORRECT",Toast.LENGTH_LONG).show();
                selected = utililities.getRandomClasses();
                String z = utililities.getCharsFromIndex(utililities.getIndexOfClass(selected));
                resText.setText(z);
                drawModel.clear();
                drawView.reset();
                drawView.invalidate();
            }else{
                Toast.makeText(getApplicationContext(),"INCORRECT",Toast.LENGTH_LONG).show();
            }
            /*String[] s = text.split("\n");
            String[] a = s[0].split(",");
            int keras = Integer.parseInt(a[0].replace("","Keras: "));
            a = s[1].split(",");
            int tf = Integer.parseInt(a[0].replace("TensorFlow: ",""));
            if(keras==selected||tf==selected){
                selected = utililities.getRandomClasses();
                String z = utililities.getCharsFromIndex(utililities.getIndexOfClass(selected));
                resText.setText(z);
                Toast.makeText(getApplicationContext(),"CORRECT",Toast.LENGTH_LONG).show();
            }else{
                Toast.makeText(getApplicationContext(),"INCORRECT",Toast.LENGTH_LONG).show();
            }*/
        }
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {

        int action = event.getAction() & MotionEvent.ACTION_MASK;

        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;

        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;

        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }



    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();

        drawView.calcPos(mLastX, mLastY, mTmpPiont);
        float lastConvX = mTmpPiont.x;
        float lastConvY = mTmpPiont.y;
        drawModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        drawView.calcPos(x, y, mTmpPiont);
        float newConvX = mTmpPiont.x;
        float newConvY = mTmpPiont.y;
        drawModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        drawView.invalidate();
    }

    private void processTouchUp() {
        drawModel.endLine();
    }


    @Override
    protected void onResume() {
        drawView.onResume();
        super.onResume();
    }

    @Override
    protected void onPause() {
        drawView.onPause();
        super.onPause();
    }

    private void loadModel() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "Keras",
                                    "frozen_model.pb", "Label.txt", PIXEL_WIDTH,
                                    "conv2d_1_input", "dense_2/Softmax", false));
                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "TensorFlow",
                                    "model.pb", "Label.txt", PIXEL_WIDTH,
                                    "conv2d_1_input", "dense_2/Softmax", false));

                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }


}
