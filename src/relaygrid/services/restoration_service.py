from __future__ import annotations
from dataclasses import dataclass

from relaygrid.restoration.sequence import (
    CandidateStep,
    LoadStep,
    dynamic_sequence,
    rank_steps,
)
from relaygrid.loads.cold_load import ColdLoadModel, CustomerType, LoadObservation
from relaygrid.loads.priority import LoadBlock, select_to_serve
from relaygrid.der.islanding import Resource, assess_island

@dataclass(frozen=True)
class RestorationRequest:
    capacity_kw: float
    load_blocks: tuple[LoadBlock,...]
    switching_candidates: tuple[CandidateStep,...]
    resources: tuple[Resource,...]=()

class RestorationService:
    def plan(self,request: RestorationRequest):
        served=select_to_serve(list(request.load_blocks),request.capacity_kw)
        ranked=rank_steps(list(request.switching_candidates))
        island_load=sum(served.values())
        island=assess_island(island_load,list(request.resources)) if request.resources else None
        return {"served":served,"switching":[s.branch_id for s in ranked],"island":island}

    def plan_dynamic(
        self,
        request: RestorationRequest,
        *,
        observations: list[LoadObservation] | tuple[LoadObservation, ...] = (),
        outage_hours: float = 1.0,
        step_interval_hours: float = 0.25,
        model: ColdLoadModel | None = None,
    ):
        """Sequence restoration with the adaptive cold-load model.

        Each load block is restored as one step; demand is recomputed from
        outage duration, customer type, and the observed restoration curve.
        Executed steps are never rolled back.  With no observations the
        conservative type-based profile is used and the basis is recorded
        per step.
        """
        steps = [
            LoadStep(
                block_id=b.block_id,
                kw=b.kw,
                priority=b.priority,
                customer_type=getattr(b, "customer_type", CustomerType.UNKNOWN),
                min_service_kw=b.min_service_kw,
            )
            for b in request.load_blocks
        ]
        return dynamic_sequence(
            steps,
            request.capacity_kw,
            observations=observations,
            outage_hours=outage_hours,
            step_interval_hours=step_interval_hours,
            model=model,
        )

    def _conservative_demand(
        self,
        b: LoadBlock,
        model: ColdLoadModel,
        outage_hours: float,
    ) -> float:
        ctype = getattr(b, "customer_type", CustomerType.UNKNOWN)
        d, _, _ = model.demand_kw(b.block_id, b.kw, ctype, outage_hours, 0.0, ())
        return d

    def unserved_kw(self,request: RestorationRequest,*,outage_hours: float=1.0,model: ColdLoadModel | None=None):
        model = model or ColdLoadModel()
        plan=self.plan(request)
        demand=sum(self._conservative_demand(b,model,outage_hours) for b in request.load_blocks)
        return max(0.0,demand-sum(plan["served"].values()))

    def critical_served_ratio(self,request: RestorationRequest,*,outage_hours: float=1.0,model: ColdLoadModel | None=None):
        model = model or ColdLoadModel()
        plan=self.plan(request)
        critical=[b for b in request.load_blocks if b.priority>=8]
        denom=sum(self._conservative_demand(b,model,outage_hours) for b in critical)
        if denom<=0: return 1.0
        num=sum(plan["served"].get(b.block_id,0.0) for b in critical)
        return num/denom
