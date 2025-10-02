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
            
            if align_token == "C":
                op_type = OpType.MATCH
            if align_token == "S":
                op_type = OpType.SUBSTITUTE
            if align_token == "I":
                op_type = OpType.INSERT
            if align_token == "D":
                op_type = OpType.DELETE

            alignment = Alignment(
                op_type=op_type,
                ref=ref_token,
                hyp=hyp_token,
            )
            alignments.append(alignment)

        return alignments
