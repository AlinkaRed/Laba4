package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.scene.text.TextFlow;
import org.example.processoremulatorfx.*;
import org.example.processoremulatorfx.BModels.BCpuModel;
import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

public class MemoryViewController implements IObserver{
  CpuModel cpuModel = BCpuModel.build();

  @FXML
  void initialize()
  {
    cpuModel.addObserver(this);
  }

  @FXML
  VBox Mem;

  @Override
  public void event(CpuModel c){
    int[] mem = c.getMemory();
    TextFlow textFlow = new TextFlow();
    textFlow.setLineSpacing(-5);
    int columns = 5;
    int indexWidth = 2;
    int valueWidth  = 2;

    for (int i = 0; i < 40; i++) {
      // Форматирование индекса и значения с фиксированной шириной, убраны квадратные скобки
      String index = String.format("%" + indexWidth + "d", i);
      String value = String.format("%" + valueWidth + "d", mem[i]);

      // Создаем текстовые объекты для индекса, двоеточия и значения
      Text indexText = new Text(index + ": ");
      Text valueText = new Text(value);
      Text space = new Text("  ");  // Добавляем пробелы для разделения колонок

      // Устанавливаем красный цвет для ненулевых значений
      if (mem[i] != 0) {
        valueText.setFill(Color.RED);
      }

      // Применяем моноширинный шрифт для всех текстовых элементов
      Font monoFont = Font.font("monospaced", 8);
      indexText.setFont(monoFont);
      valueText.setFont(monoFont);
      space.setFont(monoFont);

      // Добавляем тексты в TextFlow
      textFlow.getChildren().addAll(indexText, valueText, space);

      // Добавляем перенос строки каждые 5 элементов
      if ((i + 1) % columns == 0) {
        textFlow.getChildren().add(new Text("\n"));
      }
    }
    VBox root = new VBox(textFlow);
    Mem.getChildren().clear();
    Mem.getChildren().add(root);
  }

  @Override
  public void event(ExecuterModel e) {}

  @Override
  public void event(ProgramModel m) {}
}
