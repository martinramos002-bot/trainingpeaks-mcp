"""Tests for equipment tools."""

from unittest.mock import AsyncMock, patch

import pytest

from tp_mcp.client.http import APIResponse
from tp_mcp.tools.equipment import (
    tp_create_equipment,
    tp_delete_equipment,
    tp_get_equipment,
    tp_update_equipment,
)

MOCK_EQUIPMENT = [
    {"equipmentId": 1, "name": "Tarmac SL7", "equipmentType": 1, "brand": "Specialized",
     "model": "SL7", "distance": 5000000, "startingDistance": 0, "maxDistance": 0,
     "retired": False, "isDefault": True},
    {"equipmentId": 2, "name": "Vaporfly", "equipmentType": 2, "brand": "Nike",
     "model": "Vaporfly 3", "distance": 500000, "startingDistance": 0, "maxDistance": 800000,
     "retired": False, "isDefault": False},
]


class TestGetEquipment:
    @pytest.mark.asyncio
    async def test_returns_formatted_list(self):
        response = APIResponse(success=True, data=MOCK_EQUIPMENT)
        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_get_equipment()

        assert result["count"] == 2
        assert result["equipment"][0]["distance_km"] == 5000.0
        assert result["equipment"][0]["type"] == "bike"
        assert result["equipment"][1]["type"] == "shoe"

    @pytest.mark.asyncio
    async def test_filter_by_type(self):
        response = APIResponse(success=True, data=MOCK_EQUIPMENT)
        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_get_equipment(type="bike")

        assert result["count"] == 1
        assert result["equipment"][0]["name"] == "Tarmac SL7"


class TestCreateEquipment:
    @pytest.mark.asyncio
    async def test_create_appends_with_null_id(self):
        get_response = APIResponse(success=True, data=MOCK_EQUIPMENT.copy())
        put_response = APIResponse(success=True, data=None)

        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_response)
            mock_instance.put = AsyncMock(return_value=put_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_create_equipment(name="New Bike", type="bike", brand="Canyon")

        assert result["success"] is True
        put_payload = mock_instance.put.call_args[1]["json"]
        assert len(put_payload) == 3  # 2 existing + 1 new
        new_item = put_payload[-1]
        assert new_item["equipmentId"] is None
        assert new_item["name"] == "New Bike"
        assert new_item["equipmentType"] == 1

    @pytest.mark.asyncio
    async def test_create_converts_km_to_metres(self):
        get_response = APIResponse(success=True, data=[])
        put_response = APIResponse(success=True, data=None)

        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_response)
            mock_instance.put = AsyncMock(return_value=put_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_create_equipment(
                name="Used Bike", type="bike", starting_distance_km=2000.0,
            )

        assert result["success"] is True
        new_item = mock_instance.put.call_args[1]["json"][-1]
        assert new_item["startingDistance"] == 2000000

    @pytest.mark.asyncio
    async def test_create_bike_with_wheels(self):
        get_response = APIResponse(success=True, data=[])
        put_response = APIResponse(success=True, data=None)

        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_response)
            mock_instance.put = AsyncMock(return_value=put_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_create_equipment(
                name="TT Bike", type="bike", wheels="Zipp 808", crank_length_mm=172.5,
            )

        assert result["success"] is True
        new_item = mock_instance.put.call_args[1]["json"][-1]
        assert new_item["wheels"] == "Zipp 808"
        assert new_item["crankLength"] == 172.5

    @pytest.mark.asyncio
    async def test_create_shoe_rejects_bike_fields(self):
        result = await tp_create_equipment(
            name="Shoe", type="shoe", wheels="Not valid",
        )
        assert result["isError"] is True
        assert "bike" in result["message"].lower()


class TestUpdateEquipment:
    @pytest.mark.asyncio
    async def test_update_merges(self):
        get_response = APIResponse(success=True, data=MOCK_EQUIPMENT.copy())
        put_response = APIResponse(success=True, data=None)

        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_response)
            mock_instance.put = AsyncMock(return_value=put_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_update_equipment(equipment_id="1", name="Updated Name")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_retire_sets_date(self):
        equipment = [{"equipmentId": 1, "name": "Old", "equipmentType": 1, "retired": False}]
        get_response = APIResponse(success=True, data=equipment)
        put_response = APIResponse(success=True, data=None)

        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_response)
            mock_instance.put = AsyncMock(return_value=put_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_update_equipment(equipment_id="1", retired=True)

        assert result["success"] is True
        updated = mock_instance.put.call_args[1]["json"][0]
        assert updated["retired"] is True
        assert "retiredDate" in updated


class TestDeleteEquipment:
    @pytest.mark.asyncio
    async def test_delete_removes_from_array(self):
        get_response = APIResponse(success=True, data=MOCK_EQUIPMENT.copy())
        put_response = APIResponse(success=True, data=None)

        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_response)
            mock_instance.put = AsyncMock(return_value=put_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_delete_equipment("1")

        assert result["success"] is True
        remaining = mock_instance.put.call_args[1]["json"]
        assert len(remaining) == 1
        assert remaining[0]["equipmentId"] == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        get_response = APIResponse(success=True, data=MOCK_EQUIPMENT.copy())

        with patch("tp_mcp.tools.equipment.TPClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.ensure_athlete_id = AsyncMock(return_value=123)
            mock_instance.get = AsyncMock(return_value=get_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await tp_delete_equipment("999")

        assert result["isError"] is True
        assert result["error_code"] == "NOT_FOUND"
