from collections import OrderedDict


class BaseTask:
    def __init__(self):
        super().__init__()
        self.key = ""  # task's name
        self.loss_names = []  # added to the model's losses
        self.needs_D = False  # Task is GAN-based
        self.needs_lr = False  # Specify lr?
        self.needs_z = False  # Task has specific input, not A|B _real
        self.module_name = ""  # Generator decoder module
        self.lambda_key = ""  # --lambda_<self.lambda_key> in the params
        self.priority = 0  # tasks are sorted by increasing priority/complexity
        self.threshold_key = ""  # will be set in setup()
        self.target_key = None  # will be set in setup() if not specified
        self.eval_visuals_pred = False  # prediction is an image of some sort
        self.eval_visuals_target = False  # target is an image of some sort
        self.eval_acc = False  # measure accuracy for task?
        self.log_type = "acc"  # log loss or acc?
        self.output_dim = 0  # if logging acc, it is the number of potential classes
        self.loader_resize_target = True  # target data should be cropped/resized etc.
        self.loader_resize_input = True  # input data should be cropped/resized etc.
        self.loader_flip = True  # flip input and target data in loader?
        self.input_key = ""  # key in batch dict for this task's input data

    def setup(self):

        assert self.key
        assert self.key not in {"idt", "z", "fake", "rec"}
        assert self.lambda_key not in {"idt", "A", "B"}
        assert self.threshold_type in {"acc", "loss"}
        assert self.log_type in {"acc", "vis"}

        if not self.module_name:
            self.module_name = self.key

        if self.needs_lr and not self.module_name:
            raise ValueError("needs_lr and module_name")

        self.loss_names.append(f"G_A_{self.key}")
        self.loss_names.append(f"G_B_{self.key}")
        # if self.needs_D:
        #     self.loss_names.append(f"D_A_{self.key}")
        #     self.loss_names.append(f"D_B_{self.key}")

        if self.target_key is None:
            self.target_key = self.key + "_target"

        if self.log_type == "acc":
            assert self.output_dim > 0

        self.threshold_key = f"{self.key}_{self.threshold_type}_threshold"

        if not self.input_key:
            self.input_key = self.key

    def __str__(self):
        s = self.__class__.__name__ + ":\n"
        for d in dir(self):
            if not d.startswith("__"):
                attr = getattr(self, d)
                if not callable(attr):
                    s += "   {:15}: {}\n".format(d, attr)
        return s


class RotationTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.eval_acc = True
        self.eval_visuals_pred = False
        self.eval_visuals_target = False
        self.key = "rotation"
        self.lambda_key = "R"
        self.loader_flip = False
        self.loader_resize_input = True
        self.loader_resize_target = False
        self.log_type = "acc"
        self.needs_z = True
        self.output_dim = 4
        self.priority = 0
        self.threshold_type = "acc"


class JigsawTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.eval_acc = True
        self.eval_visuals_pred = False
        self.eval_visuals_target = False
        self.key = "jigsaw"
        self.lambda_key = "J"
        self.loader_flip = False
        self.loader_resize_input = True
        self.loader_resize_target = False
        self.log_type = "acc"
        self.needs_z = True
        self.output_dim = 64
        self.priority = 1
        self.threshold_type = "acc"


class DepthTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.eval_acc = False
        self.eval_visuals_pred = True
        self.eval_visuals_target = True
        self.input_key = "real"
        self.key = "depth"
        self.lambda_key = "D"
        self.log_type = "vis"
        self.needs_lr = True
        self.needs_z = False
        self.priority = 2
        self.threshold_type = "loss"


class GrayTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.eval_acc = False
        self.eval_visuals_pred = True
        self.eval_visuals_target = False
        self.key = "gray"
        self.lambda_key = "G"
        self.log_type = "vis"
        self.needs_D = True
        self.needs_lr = True
        self.needs_z = True
        self.priority = 3
        self.target_key = "real"
        self.threshold_type = "loss"


class AuxiliaryTasks:
    def __init__(self, keys=[]):
        super().__init__()
        self._index = 0
        tasks = []
        for k in keys:
            if k == "gray":
                tasks += [(k, GrayTask())]
            elif k == "rotation":
                tasks += [(k, RotationTask())]
            elif k == "depth":
                tasks += [(k, DepthTask())]
            elif k == "jigsaw":
                tasks += [(k, JigsawTask())]
            else:
                raise ValueError("Unknown Auxiliary task {}".format(k))
        tasks = sorted(tasks, key=lambda x: x[1].priority)
        self.tasks = OrderedDict(tasks)
        self.keys = list(self.tasks.keys())

        for t in self.tasks:
            self.tasks[t].setup()

    def task_before(self, k):
        if k not in self.tasks:
            return None
        index = self.keys.index(k)
        if index == 0:
            return None
        return self.keys[index - 1]

    def task_after(self, k):
        if k not in self.tasks:
            return None
        index = self.keys.index(k)
        if index >= len(self.tasks) - 1:
            return None
        return self.keys[index + 1]

    def __str__(self):
        return "AuxiliaryTasks: " + str(self.tasks)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._index += 1
            return self.tasks[self.keys[self._index - 1]]
        except IndexError:
            self._index = 0
            raise StopIteration

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.tasks[self.keys[k]]
        return self.tasks[k]
