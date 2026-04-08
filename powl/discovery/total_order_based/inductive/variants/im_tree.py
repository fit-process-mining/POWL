import os
from abc import ABC
from enum import Enum
from itertools import combinations
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.fall_through.empty_traces import EmptyTracesUVCL
from pm4py.algo.discovery.inductive.variants.imf import IMFParameters
from pm4py.objects.process_tree.obj import Operator
from pm4py.util import constants, exec_utils

from powl.discovery.total_order_based.inductive.base_case.factory import BaseCaseFactory
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory
from powl.discovery.total_order_based.inductive.fall_through.empty_traces import (
    POWLEmptyTracesUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.factory import (
    FallThroughFactory,
)
from powl.discovery.total_order_based.inductive.utils.filtering import (
    filter_most_frequent_variants,
    filter_most_frequent_variants_with_decreasing_factor,
    FILTERING_THRESHOLD,
    FILTERING_TYPE,
    FilteringType,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.discovery.total_order_based.inductive.modeling import (
    InductiveModel,
    build_model,
)
from powl.general_utils.dfg_frequency_filtering import filter_dfg_noise_keep_activities_and_repair
from powl.objects.tagged_powl.base import TaggedPOWL

T = TypeVar("T", bound=IMDataStructureUVCL)


class Parameters(Enum):
    MULTIPROCESSING = "multiprocessing"


class IMBasePOWL(ABC, Generic[T]):
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        if parameters is None:
            parameters = {}

        enable_multiprocessing = exec_utils.get_param_value(
            Parameters.MULTIPROCESSING,
            parameters,
            constants.ENABLE_MULTIPROCESSING_DEFAULT,
        )

        if enable_multiprocessing:
            from multiprocessing import Manager, Pool

            self._pool = Pool(os.cpu_count() - 1)
            self._manager = Manager()
            self._manager.support_list = []
        else:
            self._pool = None
            self._manager = None

    def instance(self) -> POWLDiscoveryVariant:
        return POWLDiscoveryVariant.TREE

    def empty_traces_cut(self) -> Type[EmptyTracesUVCL]:
        return POWLEmptyTracesUVCL

    def enable_dfg_fall_through(self) -> bool:
        return False

    def apply(
        self,
        obj: T,
        parameters: Optional[Dict[str, Any]] = None,
        second_iteration: bool = False,
    ) -> TaggedPOWL:

        noise_threshold = exec_utils.get_param_value(
            IMFParameters.NOISE_THRESHOLD, parameters, 0.0
        )

        empty_traces = self.empty_traces_cut().apply(obj, parameters)
        if empty_traces is not None:
            number_original_traces = sum(y for y in obj.data_structure.values())
            number_filtered_traces = sum(
                y for y in empty_traces[1][-1].data_structure.values()
            )

            if (
                number_original_traces - number_filtered_traces
                > noise_threshold * number_original_traces
            ):
                return self._recurse(empty_traces[0], empty_traces[1], parameters)
            else:
                obj = empty_traces[1][-1]

        powl = self.apply_base_cases(obj, parameters)
        if powl is not None:
            return powl

        cut = self.find_cut(obj, parameters)
        if cut is not None:
            powl = self._recurse(cut[0], cut[1], parameters=parameters)

        if powl is not None:
            return powl

        if FILTERING_TYPE in parameters.keys():
            filtering_type = parameters[FILTERING_TYPE]
            if filtering_type not in FilteringType:
                raise KeyError("Invalid FILTERING_TYPE: " + str(filtering_type))

            if filtering_type is FilteringType.DFG_FREQUENCY:
                if not second_iteration:
                    noise_threshold = exec_utils.get_param_value(
                        IMFParameters.NOISE_THRESHOLD, parameters, 0.0
                    )
                    filtered_ds = filter_dfg_noise_keep_activities_and_repair(obj, noise_threshold)
                    tree = self.apply(
                        filtered_ds, parameters=parameters, second_iteration=True
                    )
                    if tree is not None:
                        return tree

            elif filtering_type is FilteringType.DYNAMIC:
                filtered_log = filter_most_frequent_variants(obj.data_structure)
                if len(filtered_log.data_structure) > 0:
                    return self.apply(filtered_log, parameters=parameters)

            elif filtering_type is FilteringType.DECREASING_FACTOR:
                if FILTERING_THRESHOLD in parameters.keys():
                    t = parameters[FILTERING_THRESHOLD]
                    if isinstance(t, float) and 0 <= t < 1:
                        t = [t]
                    if isinstance(t, list):
                        for factor in t:
                            if factor > 0:
                                filtered_log = filter_most_frequent_variants_with_decreasing_factor(
                                    obj.data_structure, decreasing_factor=factor
                                )
                                if len(filtered_log.data_structure) == 0:
                                    break
                                elif len(filtered_log.data_structure) < len(
                                    obj.data_structure
                                ):
                                    return self.apply(
                                        filtered_log, parameters=parameters
                                    )
                    else:
                        raise KeyError("Invalid filtering threshold!")
            else:
                raise KeyError("Invalid filtering type!")

        ft = self.fall_through(obj, parameters)
        return self._recurse(ft[0], ft[1], parameters=parameters)

    def apply_base_cases(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[TaggedPOWL]:
        return BaseCaseFactory.apply_base_cases(obj, parameters=parameters)

    def find_cut(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        return CutFactory.find_cut(obj, parameters=parameters)

    def fall_through(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[InductiveModel, List[T]]:
        return FallThroughFactory.fall_through(
            obj,
            self._pool,
            self._manager,
            enable_dfg_fall_through=self.enable_dfg_fall_through(),
            parameters=parameters,
        )

    def _recurse(
        self,
        powl: InductiveModel,
        objs: List[T],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        children = [self.apply(obj, parameters=parameters) for obj in objs]
        return build_model(powl, children)
