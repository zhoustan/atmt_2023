from .beam import BeamSearch


class AdaptedBeamSearch(BeamSearch):
    """ Defines the Adapted beam search object for a single input sentence. """

    def __init__(self, beam_size, max_len, pad, gamma):
        super(AdaptedBeamSearch, self).__init__(beam_size, max_len, pad)
        self.gamma = gamma

    def add(self, score, node, sibling_rank):
        """ Add a penalized term on score """
        penalized_score = score - self.gamma * sibling_rank
        super(AdaptedBeamSearch, self).add(penalized_score, node)

    def add_final(self, score, node, sibling_rank):
        """ Add a penalized term on score that ended in EOS (= finished sentence) """
        # ensure all node paths have the same length for batch ops
        penalized_score = score - self.gamma * sibling_rank
        super(AdaptedBeamSearch, self).add_final(penalized_score, node)
