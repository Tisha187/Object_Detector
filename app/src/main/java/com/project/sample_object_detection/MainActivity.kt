package com.project.sample_object_detection

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.media.Image
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.project.sample_object_detection.ml.AutoModel1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    val paint = Paint()

    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap
    lateinit var button: Button
    lateinit var model: AutoModel1
    lateinit var labels : List<String>

    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)


    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        labels = FileUtil.loadLabels(this, "labels.txt.txt")
        model = AutoModel1.newInstance(this)


        // to open the gallery and select image type
        val intent = Intent()
        intent.setType("image/*")             // type of object to access.
        intent.setAction(Intent.ACTION_GET_CONTENT)



        button = findViewById(R.id.btn)
        imageView = findViewById(R.id.imagev)


        button.setOnClickListener{
            // to get the image from the gallery
            startActivityForResult(intent,101)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == 101){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            get_predictions()
        }
    }

     fun get_predictions() {

// Creates inputs for reference.
         var image = TensorImage.fromBitmap(bitmap)
         image = imageProcessor.process(image)

// Runs model inference and gets result.
         val outputs = model.process(image)
         val locations = outputs.locationsAsTensorBuffer.floatArray
         val classes = outputs.classesAsTensorBuffer.floatArray
         val scores = outputs.scoresAsTensorBuffer.floatArray
         val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer

         var mutable = bitmap.copy(Bitmap.Config.ARGB_8888,true)
         val canvas = Canvas(mutable)



//         scale the value to the image
         var h = mutable.height
         var w = mutable.width


         paint.textSize = h/15f
         paint.strokeWidth = h/85f
         var x = 0
         scores.forEachIndexed { index, fl ->
             if(fl > 0.5){
                 var x = index
                 x *= 4
                 paint.setColor(colors.get(index))
                 paint.style = Paint.Style.STROKE
                 canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
                 paint.style = Paint.Style.FILL
                 canvas.drawText( labels[classes.get(index).toInt()] + " " +fl.toString(), locations.get(x+1)*w, locations.get(x)*h, paint)
             }

         }

         imageView.setImageBitmap(mutable)

    }


}