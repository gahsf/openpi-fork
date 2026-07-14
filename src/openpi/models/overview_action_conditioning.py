import flax.struct as struct

from openpi.shared import array_typing as at


@struct.dataclass
class PrefixEmbeddings:
    tokens: at.Float[at.Array, "b s emb"]
    input_mask: at.Bool[at.Array, "b s"]
    ar_mask: at.Bool[at.Array, " s"]
    image_lengths: tuple[int, ...] = struct.field(pytree_node=False)
    language_length: int = struct.field(pytree_node=False)
