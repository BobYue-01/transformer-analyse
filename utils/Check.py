from typing import Optional
import math
import torch

control_codes = {
    'reset': 0,
    'bold': 1,
    'dim': 2,
    'italic': 3,
    'underline': 4,
    'reverse': 7,
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'magenta': 35,
    'cyan': 36,
    'white': 37,
    'bg_black': 40,
    'bg_red': 41,
    'bg_green': 42,
    'bg_yellow': 43,
    'bg_blue': 44,
    'bg_magenta': 45,
    'bg_cyan': 46,
    'bg_white': 47,
}

def fmt_str(message: str, *formats: str) -> str:
    start = '\x1b[' + ';'.join([str(control_codes[i]) for i in formats]) + 'm'
    return start + message + '\x1b[0m'

def fail_str(message: str) -> str:
    return fmt_str(message, 'bold', 'red')

def pass_str(message: str) -> str:
    return fmt_str(message, 'bold', 'green')

def warn_str(message: str) -> str:
    return fmt_str(message, 'bold', 'yellow')

def info_str(message: str) -> str:
    return fmt_str(message, 'bold', 'blue')

def hint_str(message: str) -> str:
    return fmt_str(message, 'dim')


class Check:
    def __init__(self) -> None:
        self.count = 0
        self.history: list[tuple[str, str, bool, float, float]] = []
        self.print_val = True
        self.all_rel_tol = None
        self.all_abs_tol = None

    def hide_val(self) -> None:
        self.print_val = False

    def show_val(self) -> None:
        self.print_val = True

    def set_all_tol(self,
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None
    ) -> None:
        self.all_rel_tol = rel_tol
        self.all_abs_tol = abs_tol

    def check_eq(
        self,
        a_str: str,
        b_str: str,
        global_vars: dict = globals(),
        local_vars: dict = locals(),
        raise_exception: bool = False,
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None,
    ) -> bool:
        if rel_tol is None:
            if self.all_rel_tol is not None:
                rel_tol = self.all_rel_tol
            else:
                rel_tol = 1e-5

        if abs_tol is None:
            if self.all_abs_tol is not None:
                abs_tol = self.all_abs_tol
            else:
                abs_tol = 1e-8

        print(info_str(f"# {self.count} [ Test ] {a_str} ?= {b_str}"))
        a = eval(a_str, global_vars, local_vars)
        b = eval(b_str, global_vars, local_vars)

        if self.print_val:
            with torch._tensor_str.printoptions(precision=4, sci_mode=True):
                print(hint_str(f"=== {a_str} ==="))
                print(f"{a}")
                print(hint_str(f"=== {b_str} ==="))
                print(f"{b}")

        if isinstance(a, (torch.Tensor, torch.nn.parameter.Parameter)) \
        and isinstance(b, (torch.Tensor, torch.nn.parameter.Parameter)):
            try:
                equal = torch.allclose(a, b, rtol=rel_tol, atol=abs_tol)
            except RuntimeError:
                equal = False
            if not equal:
                if a.shape != b.shape:
                    print(fail_str(f"Shapes of {a_str} and {b_str} are different"))
                    print(fail_str(f"{a_str}.shape: {a.shape}"))
                    print(fail_str(f"{b_str}.shape: {b.shape}"))
                else:
                    diff = torch.abs(a - b)
                    mean_diff = torch.mean(diff, dim=-1, keepdim=True)
                    # 如果 diff 每行中的值都分别相等
                    if torch.allclose(mean_diff, diff, rtol=rel_tol, atol=abs_tol):
                        print(warn_str(f"Mean abs diff: {mean_diff.abs().mean()}"))
                    else:
                        max_diff, max_index = torch.max(diff.flatten(), 0)
                        # 将展平索引转换为多维索引
                        max_index_unraveled = torch.unravel_index(max_index, diff.shape)

                        with torch._tensor_str.printoptions(precision=4, sci_mode=True):
                            print(warn_str("\n".join([
                                # f"a: {a}",
                                # f"b: {b}",
                                # f"",
                                f"Max diff: {max_diff}",
                                f"Location: {[i.item() for i in max_index_unraveled]}",
                                f"{a_str}: {a[max_index_unraveled]}",
                                f"{b_str}: {b[max_index_unraveled]}",
                                # f"",
                            ])))
        elif type(a) != type(b):
            assertion_str = fail_str(
                f"# {self.count} [ Fail ] {a_str} and {b_str} have different types\n" +
                f"{type(a) = } \n" +
                f"{type(b) = }"
            )
            if raise_exception:
                raise AssertionError(assertion_str)
            else:
                print(assertion_str)
            self.history.append((a_str, b_str, False, math.nan, math.nan))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            equal = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        else:
            equal = a == b

        if not equal:
            if raise_exception:
                raise AssertionError(
                    fail_str(f"# {self.count} [ Fail ] {a_str} != {b_str}")
                )
            else:
                print(fail_str(f"# {self.count} [ Fail ] {a_str} != {b_str}"))
            self.history.append((a_str, b_str, False, rel_tol, abs_tol))
        else:
            print(pass_str(f"# {self.count} [ Pass ] {a_str} == {b_str}"))
            self.history.append((a_str, b_str, True, rel_tol, abs_tol))

        self.count += 1
        print()
        return equal

    def summary(self):
        print(info_str(f"==== < Summary > ===="))
        for i, (a_str, b_str, result, rel_tol, abs_tol) in enumerate(self.history):
            if result:
                print(
                    pass_str(f"# {i} [ Pass ]"), f"{a_str} == {b_str}",
                    hint_str(f"(rel_tol={rel_tol}, abs_tol={abs_tol})")
                )
            else:
                print(
                    fail_str(f"# {i} [ Fail ]"), f"{a_str} != {b_str}",
                    hint_str(f"(rel_tol={rel_tol}, abs_tol={abs_tol})")
                )
        print(f"-------------------")
        total = len(self.history)
        pass_count = sum([1 for _, _, result, _, _ in self.history if result])
        print(info_str(f"({pass_count}/{total}) [") + "".join([
            pass_str("=") if result else
            fail_str("X") for _, _, result, _, _ in self.history
        ]) + info_str("]"))
        print(info_str(f"==== </Summary > ===="))
        print()

if __name__ == "__main__":
    check = Check()

    a = 1
    b = 1

    check.check_eq('a', 'b')

    check.summary()

    c = 1
    d = 2

    check.check_eq('c', 'd')

    x = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])
    y = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])

    check.check_eq('x', 'y')

    u = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    v = torch.tensor([
        [1, 2, 3],
        [4, 6, 8]
    ])

    check.check_eq('u', 'v')

    e = 1
    f = 1

    check.check_eq('e', 'f', raise_exception=True)

    check.summary()

    g = 1
    h = 2

    check.check_eq('g', 'h', raise_exception=True)