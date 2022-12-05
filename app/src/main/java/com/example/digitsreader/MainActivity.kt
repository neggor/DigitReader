package com.example.digitsreader

import android.Manifest
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


class MainActivity: AppCompatActivity(){
    private val cameraRequest = 1888
    lateinit var imageView: ImageView
    lateinit var results: TextView
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
        fun classifyImage(photo: Bitmap){

            // Image processing:
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(size, size, ResizeOp.ResizeMethod.BILINEAR))
                //.add(NormalizeOp(0.0F, 255.0F))
                .build()

            //var tensorImage = TensorImage(DataType.FLOAT32)
            var tensorImage = TensorImage(DataType.UINT8)
            // Get from bitmap
            tensorImage.load(photo)
            // Apply transformation
            tensorImage = imageProcessor.process(tensorImage);
            //
            // Load model
            //val model = ModelUnquant.newInstance(this)
            val model = ModelRgbPointEstimate.newInstance(this)
            //

            // Creates inputs for reference.
            // Allocate memory
            //var byteBuffer: ByteBuffer =  ByteBuffer.allocateDirect(4*size*size*3)
            // Populate it, back from tensor image to bitmap then to byte buffer
            //tensorImage.tensorBuffer

            // Allocate
            //val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, size, size, 3), DataType.FLOAT32)

            //Populate
            //inputFeature0.loadBuffer(byteBuffer)

            // Evaluate
            var outputs = model.process(tensorImage.tensorBuffer)

            model.close()

            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val confidence = outputFeature0.floatArray

            var maxPos: Int = 0
            var maxConfidence: Float = 0.0F

            for(i in confidence.indices){
                   if (confidence[i] > maxConfidence){
                    maxConfidence = confidence[i]
                    maxPos = i
                }
            }
            val my_classes = (0..9).map { e -> e.toString()}
            //val my_classes = listOf("Luise", "Mentos")
            results = findViewById(R.id.result)
            results.text = my_classes[maxPos]
            confidence_text = findViewById(R.id.confidence)
            //confidence_text.text = confidence.contentToString()
            confidence_text.text = confidence[maxPos].toString()

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