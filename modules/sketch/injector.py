from einops import rearrange
from pathlib import Path
import numpy as np
from utils.pca_visualizer import PCAVisualizer


recode_time = 201
target_Q_location = [15, 17, 21, 23]
target_hs_location = [14, 15, 17, 18]
end_time = 300
start_time = 1000


class Injector:
    def __init__(self):
        self.hidden_states_count = 0
        self.hidden_states = []
        self.Q_count = 0
        self.Q = []
        self.t = -1
        self.forward = True
        self.isSketch = False
        self.mode = "sample"
        self.attn_count = 0
        self.save_dir = Path("result/pca")

    def reset(self, t, forward=True, mode=None):
        self.t = t
        self.forward = forward
        self.hidden_states_count = 0
        self.Q_count = 0
        self.attn_count = 0
        if mode is None:
            self.mode = "forward" if forward else "sample"
        else:
            self.mode = mode

    def _save_pca(self, tensor, kind, count):
        data = tensor.detach().float().reshape(tensor.shape[0], -1).cpu().numpy()
        pca = PCAVisualizer(n_components=2)
        subdir = self.save_dir / self.mode
        subdir.mkdir(parents=True, exist_ok=True)
        fname = subdir / f"t{int(self.t):03d}_b{int(count):02d}_{kind}.png"
        pca.visualize(data, save_path=str(fname))

    def recode_hidden_states(self, hidden_states):
        if self.t == recode_time and self.hidden_states_count in target_hs_location and self.forward is True:
            print("Record hidden states at step 201")
            self.hidden_states.append(hidden_states)

    def recode_Q(self, Q):
        if self.t == recode_time and self.Q_count in target_Q_location and self.forward is True:
            print("Record Q at step 201")
            self.Q.append(Q)

    def isReplace(self):
        return self.t >= end_time and self.t <= start_time and self.forward is False

    def load_features(self, hidden_states, row_hidden_states):
        b, c, h, w = row_hidden_states.shape
        patch_window = 4
        injected_hidden_states = rearrange(
            row_hidden_states,
            'b c (h p1) (w p2) -> b c (h w) p1 p2', p1=patch_window, p2=patch_window,
        )
        first_frame_hid = rearrange(
            hidden_states.clone(),
            'b c (h p1) (w p2) -> b c (h w) p1 p2', p1=patch_window, p2=patch_window,
        )
        mean1 = injected_hidden_states.reshape(b, c, -1, patch_window, patch_window).mean(dim=(-2, -1), keepdim=True)
        var1 = injected_hidden_states.reshape(b, c, -1, patch_window, patch_window).std(dim=(-2, -1), keepdim=True)
        mean2 = first_frame_hid.reshape(b, c, -1, patch_window, patch_window).mean(dim=(-2, -1), keepdim=True)
        var2 = first_frame_hid.reshape(b, c, -1, patch_window, patch_window).std(dim=(-2, -1), keepdim=True)
        injected_hidden_states = (injected_hidden_states - mean1) / (var1 + 1e-6) * var2 + mean2
        injected_hidden_states = rearrange(
            injected_hidden_states,
            'b c (h w) p1 p2 -> b c (h p1) (w p2)',
            h=h // patch_window,
            w=w // patch_window,
        )
        hidden_states = injected_hidden_states.clone().detach()
        return hidden_states

    def replace_hidden_states(self, hidden_states):
        if self.isReplace():
            if self.hidden_states_count in target_hs_location:
                row_hidden_states = self.hidden_states[target_hs_location.index(self.hidden_states_count)]
            else:
                return hidden_states
            new_hidden_states = self.load_features(hidden_states[[1, 3]], row_hidden_states)
            hidden_states[1:2] = new_hidden_states[0:1]
            hidden_states[3:4] = new_hidden_states[1:2]
        return hidden_states

    def replace_Q(self, Q):
        if self.isReplace():
            if self.Q_count in target_Q_location:
                row_Q = self.Q[target_Q_location.index(self.Q_count)]
            else:
                return Q
            Q[1:2] = row_Q[0:1]
            Q[3:4] = row_Q[1:2]
        return Q

    def hook_hidden_states(self, hidden_states):
        if not self.isSketch:
            return hidden_states
        self.hidden_states_count += 1
        self.recode_hidden_states(hidden_states)
        hidden_states = self.replace_hidden_states(hidden_states)
        self._save_pca(hidden_states, "hidden", self.hidden_states_count)
        return hidden_states

    def hook_Q(self, Q):
        if not self.isSketch:
            return Q
        self.Q_count += 1
        self.recode_Q(Q)
        Q = self.replace_Q(Q)
        self._save_pca(Q, "q", self.Q_count)
        return Q

    def hook_attention_map(self, attn_map):
        if not self.isSketch:
            return attn_map
        self.attn_count += 1
        self._save_pca(attn_map, "attn", self.attn_count)
        return attn_map

    def break_invert(self, t):
        if t > recode_time and self.isSketch is True:
            return True
        return False


injector = Injector()

