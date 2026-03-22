<img width="1211" height="192" alt="image" src="https://github.com/user-attachments/assets/357983c5-a5bb-4065-988f-d55762f1e437" />

# geomatch
Geomatch is an architecture allowing for exact location pinpointing from a single image given a predefined area. This means that given an image taken in a city for example, Geomatch will be able to pretty confidently guess the exact street where that image was taken.
The setup is done with a sufficient amount of streetview images from the area where you would like to perform localization. I, for example, use 200k streetview images from Prague (4 images from 50k random locations). Although 50 thousand locations for a single city sounds sufficient, in my experience at least doubling this figure would be ideal for further research.

# Try it out
You can try out the system on [geomatch.josefbednar.com](https://geomatch.josefbednar.com/). Just upload a streetview image, wait ~20 seconds, and view top 30 candidate locations on a map and in Google Streeview embeds, then personally pick and confirm the exact location.
Note that the system was developed for absolutely free and is running 24/7 on a $4/month VPS with 4GB of RAM and a single core. Therefore, many optimizations and performance hacks were deployed to allow for reasonably useful and fast usage. If built for production on a graphics card this exact architecture could undoubtedly be replicated to work with a 100% accuracy with inference taking a fraction of the time it currently does.
For these reasons it currently only works for Prague. As well as that, it only works with images taken in daylight that are square (1:1). Additionally, as there are only 50k locations so some images may not be able to be localized at all (though the system will present the top 30 candidates anyway).

# How it works
The system performs 2 steps. First it uses 2048 dimension vector search to find 30 images with the most similar vector to the query image. All of these embeddings are calculated by the pre-trained [CosPlace model](https://github.com/gmberton/CosPlace).
Then it calculates the 1024 most prominent features (keypoints) of the query image using the [ALIKED model](https://github.com/Shiaoming/ALIKED) and re-ranks the 30 candidate images using a process called local feature matching with the [LightGlue model](https://github.com/cvg/lightglue). That ensures that the image in the database with the most real world points matching the query image ends up at the first place. This step is the biggest bottleneck during inference on a CPU.

# Setup
The system is set up in a few steps.
1. Create a dataset of images of locations ([use my repo](https://github.com/bednarjosef/streetview-scrape))
2. Configure and run setup_vectors.py and setup_features.py (vector db will be ~4GB and features ~25GB with INT8 quantization)
3. Initialize the Geomatcher object and call .get_ranked() to get the vector db ranking and feature refined ranking of an image.
