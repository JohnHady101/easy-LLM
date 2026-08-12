import tiktoken

class SimpleTokenizer:
    def __init__(self, model: str):
        self.encoder = tiktoken.encoding_for_model(model)

    def encode(self, text: str) -> list:
        splitted = [self.encoder.decode_single_token_bytes(token).decode("utf-8") for token in self.encoder.encode(text)]
        # adding <|endoftext|>
        splitted.append("<|endoftext|>")
        # remove spaces
        tokens = [item for item in splitted if item.strip()]
        # stripping tokens
        tokens = [item.strip() for item in tokens]

        encoded = []

        for token in tokens:
            # try to encode a single token, if it fails, byte pair it and encode the pieces
            try:
                encoded.append(self.encoder.encode_single_token(token))
            except KeyError:
                # Token is not a single token, so we need to encode it as a sequence
                encoded.extend(self.encoder.encode(token))


        # encoded = [self.encoder.encode_single_token(token) for token in tokens]

        return encoded

    def decode(self, token_ids: list) -> str:

        return " ".join(self.encoder.decode_single_token_bytes(id).decode("utf-8") for id in token_ids)