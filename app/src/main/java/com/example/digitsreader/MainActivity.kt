package com.example.digitsreader

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.digitsreader.ml.ModelRgbPointEstimate
import com.example.digitsreader.ml.ModelUnquant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import kotlin.math.min
import kotlin.math.roundToLong


class MainActivity: AppCompatActivity(){
    private val cameraRequest = 1888
    lateinit var imageView: ImageView
    lateinit var result1: TextView
    lateinit var result2: TextView
    lateinit var result3: TextView
    lateinit var prob1: TextView
    lateinit var prob2: TextView
    lateinit var prob3: TextView

    lateinit var confidence_text: TextView
    lateinit var gallery: Button
    val size: Int = 28
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (ContextCompat.checkSelfPermission(applicationContext, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_DENIED
        )
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                cameraRequest
            )
        imageView = findViewById(R.id.imageView)
        val photoButton: Button = findViewById(R.id.button)
        photoButton.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, cameraRequest)
        }

        gallery = findViewById(R.id.button2)
        gallery.setOnClickListener {
                val cameraIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
    }
        @SuppressLint("SetTextI18n")
        fun classifyImage(photo: Bitmap){

            // Image processing:
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(size, size, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            var tensorImage = TensorImage(DataType.UINT8)
            // Get from bitmap
            tensorImage.load(photo)
            // Apply transformation
            tensorImage = imageProcessor.process(tensorImage);

            // Load model
            val model = ModelRgbPointEstimate.newInstance(this)


            // Evaluate
            var outputs = model.process(tensorImage.tensorBuffer)

            model.close()

            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val confidence = outputFeature0.floatArray


            val myClasses = (0..9).sortedBy { -confidence[it] }.map { e -> e.toString()}

            result1 = findViewById(R.id.result1)
            result2 = findViewById(R.id.result2)
            result3 = findViewById(R.id.result3)

            prob1 = findViewById(R.id.prob1)
            prob2 = findViewById(R.id.prob2)
            prob3 = findViewById(R.id.prob3)

            result1.text = myClasses[0]
            result2.text = myClasses[1]
            result3.text = myClasses[2]

            confidence.sortDescending()
            prob1.text = "${(String.format("%.2f", confidence[0] * 100))}%"
            prob2.text = "${(String.format("%.2f", confidence[1] * 100))}%"
            prob3.text = "${(String.format("%.2f", confidence[2] * 100))}%"

        }
        override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
            super.onActivityResult(requestCode, resultCode, data)
            if (requestCode == cameraRequest) {
                var photo: Bitmap = data?.extras?.get("data") as Bitmap
                val dimension: Int = min(photo.width, photo.height)
                photo = ThumbnailUtils.extractThumbnail(photo, dimension, dimension)
                imageView.setImageBitmap(photo)

                photo = Bitmap.createScaledBitmap(photo, size, size, false)
                classifyImage(photo)
            }else{
                var dat = data?.data
                var photo: Bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, dat)
                imageView.setImageBitmap(photo)
                //photo = Bitmap.createScaledBitmap(photo, size, size, false)
                classifyImage(photo)
            }
        }

}