from error_align.baselines.power.power.aligner import PowerAligner as _PowerAligner
from error_align.utils import Alignment, OpType


class PowerAlign:
    """Phonetically-oriented word error alignment."""

    def __init__(
        self,
        ref: str,
        hyp: str,
    ):
        """Initialize the phonetically-oriented word error alignment with reference and hypothesis texts.

        Args:
            ref (str): The reference sequence/transcript.
            hyp (str): The hypothesis sequence/transcript.
        """
        self.aligner = _PowerAligner(
            ref=ref,
            hyp=hyp,
            lowercase=True,
            verbose=True,
            lexicon="/home/lb/repos/power-asr/lex/cmudict.rep.json",
        )

    def align(self):
        """Run the two-pass Power alignment algorithm.

        Returns:
            list[Alignment]: A list of Alignment objects.
        """
        self.aligner.align()
        widths = [
            max(len(self.aligner.power_alignment.s1[i]), len(self.aligner.power_alignment.s2[i]))
            for i in range(len(self.aligner.power_alignment.s1))
        ]
        s1_args = list(zip(widths, self.aligner.power_alignment.s1))
        s2_args = list(zip(widths, self.aligner.power_alignment.s2))
        align_args = list(zip(widths, self.aligner.power_alignment.align))

        alignments = []
        for (_, ref_token), (_, hyp_token), (_, align_token) in zip(s1_args, s2_args, align_args):

            # NOTE: The original Power alignments fail in a few edge cases, so we
            # implement a simple fix, where the op_type is based on the tokens instead.

            # if align_token == "C":
            #     op_type = OpType.MATCH
            # if align_token == "S":
            #     op_type = OpType.SUBSTITUTE
            # if align_token == "I":
            #     op_type = OpType.INSERT
            # if align_token == "D":
            #     op_type = OpType.DELETE

            if not ref_token and not hyp_token:
                continue
            elif ref_token and not hyp_token:
                op_type = OpType.DELETE
            elif not ref_token and hyp_token:
                op_type = OpType.INSERT
            elif ref_token and hyp_token:
                if ref_token.lower() == hyp_token.lower():
                    op_type = OpType.MATCH
                else:
                    op_type = OpType.SUBSTITUTE

            alignment = Alignment(
                op_type=op_type,
                ref=None if ref_token == "" else ref_token,
                hyp=None if hyp_token == "" else hyp_token,
            )
            alignments.append(alignment)

        return alignments
