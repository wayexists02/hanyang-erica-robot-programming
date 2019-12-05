import rospy
import torch
import copy
from torch import nn
from torch.autograd import Function


class Factor(nn.Module):

    def __init__(self, var_names, cards, vals):
        super(Factor, self).__init__()

        self.transition_mat = nn.Parameter(vals).view(*cards)
        self.var_names = var_names
        self.cards = cards

    def conditioning(self, var_names, val_indices):
        res_var_names = copy.deepcopy(self.var_names)
        res_cards = copy.deepcopy(self.cards)
        res_transition_mat = self.transition_mat.copy()

        for name, val_index in zip(var_names, val_indices):
            index = self.var_names.index(name)
            res_transition_mat = torch.index_select(res_transition_mat, index, torch.LongTensor([val_index]).cuda())
            
            res_index = res_var_names.index(name)
            res_var_names.pop(res_index)
            res_cards.pop(res_index)

        factor = Factor(res_var_names, res_cards, res_transition_mat).cuda()

        return factor

    def marginalize(self, var_names):
        res_var_names = copy.deepcopy(self.var_names)
        res_cards = copy.deepcopy(self.cards)
        res_transition_mat = self.transition_mat.copy()

        for name in var_names:
            index = self.var_names.index(name)
            res_transition_mat = torch.sum(res_transition_mat, dim=index, keepdim=True)

            res_index = res_var_names.index(name)
            res_var_names.pop(res_index)
            res_cards.pop(res_index)

        factor = Factor(res_var_names, res_cards, res_transition_mat).cuda()

        return factor

    def joint(self, var_names):
        res_var_names = copy.deepcopy(self.var_names)
        res_cards = copy.deepcopy(self.cards)
        res_transition_mat = self.transition_mat.copy()

        margin_var_names = [name for name in self.var_names if name not in var_names]
        factor = self.marginalize(margin_var_names)
        
        return factor

    def normalize(self):
        self.transition_mat /= torch.sum(self.transition_mat, keepdim=True)

    def check_valid(self):
        if len(self.var_names) != len(self.cards):
            rospy.logerror("INVALID factor!")
            return False

        return True

    def forward(self, var_names, conditioning=[]):
        if self.check_valid() is False:
            return None

        conditioning_var_names = list(map(lambda elem: elem[0], conditioning))
        conditioning_indices = list(map(lambda elem: elem[1], conditioning))

        factor = self.conditioning(conditioning_var_names, conditioning_indices)
        factor = factor.joint(var_names)
        factor.normalize()

        return factor

    @classmethod
    def mul(cls, factor1, factor2):
        common_var_names = [var_name for var_name in factor1.var_names if var_name in factor2.var_names]
        factor1_exclusive_vars = [var_name for var_name in factor1.var_names if var_name not in factor2.var_names]
        factor2_exclusive_vars = [var_name for var_name in factor2.var_names if var_name not in factor1.var_names]

        transition_mat = torch.zeros().cuda()

class FactorOperation(Function):

    @staticmethod
    def conditioning(ctx, transition_mat, var_names, conditioning=[]):

        res_var_names = copy.deepcopy(var_names)

        for name, val_index in conditioning:
            index = var_names.index(name)
            transition_mat = torch.index_select(transition_mat, index, torch.tensor(val_index))
            res_var_names.remove(name)

        transition_mat /= torch.sum(transition_mat, dim=-1, keepdim=True)
        transition_mat = transition_mat.squeeze()

        return transition_mat, res_var_names

    @staticmethod
    def marginalize(ctx, transition_mat, var_names, margin_var_names):

        res_var_names = copy.deepcopy(var_names)

        for name in margin_var_names:
            index = var_names.index(name)
            transition_mat = torch.sum(transition_mat, dim=index, keepdim=True)
            res_var_names.remove(name)

        transition_mat /= torch.sum(transition_mat, dim=-1, keepdim=True)
        transition_mat = transition_mat.squeeze()

        return transition_mat, res_var_names

    @staticmethod
    def get_joint(ctx, transition_mat, var_names, joint_var_names, conditioning=[]):
        transition_mat, var_names = ctx.conditioning(transition_mat, var_names, conditioning)

        margin_var_names = [name for name in var_names if name not in joint_var_names]
        transition_mat, var_names = ctx.marginalize(transition_mat, var_names, margin_var_names)

        return transition_mat, var_names

    @staticmethod
    def forward(ctx, transition_mat, var_names, joint_var_names, conditioning=[]):
        ctx.save_fot_backward(transition_mat)

        transition_mat, var_names = ctx.get_joint(transition_mat, var_names, joint_var_names, conditioning)
        return transition_mat, var_names
