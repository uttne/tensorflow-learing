# %%
import my_util as mu

# %%
all_path = mu.get_all_image_paths("./datasets/bottle_datasets/train/negative")

print(all_path)

# %%
image = mu.load_and_preprocess_image(all_path[0])
image

# %%
mu.show_image(image)

# %%
