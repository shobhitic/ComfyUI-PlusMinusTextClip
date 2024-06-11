class PlusMinusTextClip:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "clip": ("CLIP", {}),
        "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
        "negative": ("STRING", {"multiline": True, "dynamicPrompts": True})
      },
    }
  RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
  RETURN_NAMES = ("Positive Prompt", "Negative Prompt",)
  FUNCTION = "encode"
  CATEGORY = "conditioning"

  def encode(self, clip, positive, negative):
    return (self.encode_text(clip, positive), self.encode_text(clip, negative),)

  def encode_text(self, clip, text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return ([cond, {"pooled_output": pooled}],)

