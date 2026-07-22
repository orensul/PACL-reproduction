from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pycocotools.coco import COCO

from PIL import Image
import spacy
import random
import open_clip

# Cross-referenced with docs/paper_summary.txt.
#   [summary §1 Training data] -- the paper trains on GCC-3M + GCC-12M + YFCC15M image-text
#       pairs. This reimpl substitutes MS-COCO (image + caption per example); no masks/boxes are
#       used, matching §1's "No segmentation masks, bounding boxes, or patch annotations".
#   [summary §2 Prompt augmentation] -- implemented in __getitem__ (noun extraction + templates).
class CocoDataset(Dataset):
    # [summary §2] paper's exact 7 CLIP prompt templates (Appendix A.2.2), vs. this reimpl's
    # original 5. Used when paper_faithful_prompts=True.
    PAPER_TEMPLATES = [
        'itap of a {}.',
        'a bad photo of the {}.',
        'a origami {}.',
        'a photo of the large {}.',
        '{} in a video game.',
        'art of the {}.',
        'a photo of the small {}.',
    ]
    ORIGINAL_TEMPLATES = [
        'a picture of {}.',
        'itap of {}.',
        'a photograph of {}.',
        'this picture contains {}.',
        'a good photo of {}.'
    ]

    def __init__(self, root_dir, annotation_file, apply_transform=False, img_size=400,
                 paper_faithful_prompts=False):

        # chunk for original COCO dataloader
        self.root_dir = root_dir
        self.train_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.RandomHorizontalFlip(0.5),
                        T.ColorJitter(brightness=.2, hue=.1),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.val_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.apply_transform = apply_transform
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # [summary §2: "wrap them with one of seven CLIP prompt templates"] -- the templates.
        self.paper_faithful_prompts = paper_faithful_prompts
        self.template = self.PAPER_TEMPLATES if paper_faithful_prompts else self.ORIGINAL_TEMPLATES
        self.nlptk = spacy.load("en_core_web_sm")

        # chunk for tokenization
        self.open_clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')


    def __len__(self):
        # paper_faithful_prompts mode emits TWO training examples per image per epoch (the
        # original caption AND a templated noun phrase, see __getitem__) instead of one.
        return len(self.ids) * 2 if self.paper_faithful_prompts else len(self.ids)

    def _load_image(self, img_info):
        img_path = f"{self.root_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        return self.train_transform(image) if self.apply_transform else self.val_transform(image)

    def _noun_phrase_text(self, caption):
        processed_text = self.nlptk(caption)
        all_noun_phrases = [chunk.text.lower() for chunk in processed_text.noun_chunks]
        if len(all_noun_phrases) == 0:
            return caption
        random_noun_phrase = random.choice(all_noun_phrases)
        random_template = random.choice(self.template)
        return random_template.format(random_noun_phrase)

    def __getitem__(self, index):
        coco = self.coco

        if self.paper_faithful_prompts:
            # [summary §2: "These prompts are added alongside the original caption."] -- literally
            # two separate training examples per image: the original caption (variant 0) and a
            # templated noun phrase (variant 1), rather than a 50/50 substitute for one slot.
            img_idx, variant = divmod(index, 2)
            img_id = self.ids[img_idx]
            caption = coco.imgToAnns[img_id][0]['caption']
            img_info = coco.loadImgs(img_id)[0]
            image = self._load_image(img_info)
            text = caption if variant == 0 else self._noun_phrase_text(caption)
            tokenized_phrase = self.open_clip_tokenizer(text).squeeze()
            return image, tokenized_phrase

        # Original behavior: ONE text per image per step, half the time the original caption,
        # half the time a templated noun phrase -- see README "Differences from the paper".
        img_id = self.ids[index]
        caption = coco.imgToAnns[img_id][0]['caption'] # a python string
        img_info = coco.loadImgs(img_id)[0]
        image = self._load_image(img_info)

        nounphrase_or_full_caption = random.choice([0, 1])
        if nounphrase_or_full_caption == 0:
            single_noun_phrase_per_img = self._noun_phrase_text(caption)
        else:
            single_noun_phrase_per_img = caption

        tokenized_phrase = self.open_clip_tokenizer(single_noun_phrase_per_img).squeeze()
        return image, tokenized_phrase

# # # Define the paths to the dataset and annotations
# data_dir = "/home/Dataset/Visual_Recognition/MSCOCO/val2017/"
# annotation_file = "/home/Dataset/Visual_Recognition/MSCOCO/annotations/captions_val2017.json"

# # # Create the dataset and dataloader
# coco_dataset = CocoDataset(data_dir, annotation_file)
# coco_loader = DataLoader(coco_dataset, batch_size=64, shuffle=True)

# for image, caption in coco_loader:
#     print(image.shape, caption.shape)
#     break
