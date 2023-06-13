# SwatchTransformer

Generative Color Swatch Transformer:

Using the transformer architecture to model color palettes trained on palettes from the excellent [palettable](https://pypi.org/project/palettable/) package. 

This project was cool, to me, because it is a straightforward seq2seq problem statement that would allow me to understnad and play with transformers using a simple modality (sequences of 3-vectors). So far, it does a decent job at producing qualitative color palettes, but without positonal encoding and palette type encoding it struggles with producing sequential and divergent palettes.

I've included a sample gif below, and you can try it out at steffanic.github.io/SwatchTransformer/, too!

![image](https://user-images.githubusercontent.com/38746732/230979805-b4c86452-c24f-44b8-b39e-1d7c5f31fe1d.gif)


